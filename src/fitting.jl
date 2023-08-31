function grad_logprob(model, input, target, shock, params)
    grad_logprob!(
        model.agent,
        zero(model.agent),
        [0.0],
        [1.0],
        input,
        zero(input),
        target,
        zero(target),
        shock,
        zero(shock),
        params,
        zero(params),
    )
end
# warning: Base method extended with types FlyRL doesn't own.
function Base.zero(x::AbstractVector{<:AbstractVector})
    [zero(xi) for xi in x]
end
zero!(x::AbstractArray) = x .= 0
function zero!(x::AbstractVector{<:AbstractVector})
    for xi in x
        xi .= 0
    end
    x
end
function grad_logprob!(
    agent,
    dagent,
    res,
    dres,
    input,
    dinput,
    target,
    dtarget,
    shock,
    dshock,
    params,
    dparams,
)
    GC.enable(false)
    Enzyme.autodiff(
        Enzyme.Reverse,
        logprob!,
        Duplicated(res, dres),
        Duplicated(agent, dagent),
        Duplicated(input, dinput),
        Duplicated(target, dtarget),
        Duplicated(shock, dshock),
        Duplicated(params, dparams),
    )
    GC.enable(true)
#     GC.gc()
    res[], dparams
end

abstract type AbstractOptimiser end
mutable struct Adam <: AbstractOptimiser
    eta::Float64
    beta::Tuple{Float64,Float64}
    epsilon::Float64
    state::IdDict{Any,Any}
end
Adam(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = 1e-8) = Adam(η, β, ϵ, IdDict())
Adam(η::Real, β::Tuple, state::IdDict) = Adam(η, β, 1e-8, state)

function apply!(o::Adam, x, Δ)
    η, β = o.eta, o.beta

    mt, vt, βp = get!(o.state, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end::Tuple{typeof(x),typeof(x),Vector{Float64}}

    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    @. Δ = mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
    βp .= βp .* β

    return Δ
end
mutable struct Descent <: AbstractOptimiser
    eta::Float64
end
function apply!(o::Descent, x, Δ)
    Δ .*= o.eta
end

cvec(x::ComponentArray, ::Any) = x
cvec(x, ax) = ComponentArray(x, ax)
fix!(x, ::Nothing) = x
function fix!(x, fixed)
    for (k, v) in pairs(fixed)
        setproperty!(x, k, v)
    end
    x
end
append_fixed(x, ::Nothing) = x
append_fixed(x, fixed) = ComponentArray(x; fixed...)
remove_fixed(x, ::Nothing) = x
function remove_fixed(x, fixed)
    ComponentArray(
        NamedTuple(k => getproperty(x, k) for k in setdiff(keys(x), keys(fixed))),
    )
end
function _batches(T, batchsize)
    p = randperm(T)
    [p[i*batchsize+1:min(T, (i+1)*batchsize)] for i in 0:T÷batchsize]
end
_batches(T, ::Nothing) = [1:T]
function gradient_func(logp, dlogp,
                       agent, dagent,
                       input, dinput,
                       target, dtarget,
                       shock, dshock,
                       ax, fixed; batchsize = nothing)
    batches = _batches(length(target), batchsize)
    batchid = 1
    logps = Array{Union{Float64,Missing}}(undef, length(batches))
    (f, dparams, params) -> begin
        params = append_fixed(cvec(params, ax), fixed)
        if dparams !== nothing && length(dparams) > 0
            d = append_fixed(cvec(dparams, ax), fixed)
            logp[] = 0.0
            dlogp[] = 1.0
            zero!(dagent)
            zero!(dinput)
            zero!(dtarget)
            zero!(dshock)
            zero!(d)
            idxs = batches[batchid]
            grad_logprob!(
                agent,
                dagent,
                logp,
                dlogp,
                view(input, idxs),
                view(dinput, idxs),
                view(target, idxs),
                view(dtarget, idxs),
                view(shock, idxs),
                view(dshock, idxs),
                params,
                d,
            )
            dparams .= -remove_fixed(d, fixed)
        elseif f !== nothing
            logprob!(logp, agent, input, target, shock, params)
        end
        logps[batchid] = -logp[]
        batchid = batchid % length(batches) + 1
        if f !== nothing
            return _replace_missing(logps)
        end
    end
end
function _replace_missing(logps)
    ret = 0.
    for lp in logps
        if ismissing(lp)
            ret += 5*minimum(skipmissing(logps)) # underestimate missing batches
        else
            ret += lp
        end
    end
    ret/length(logps)
end
function wrap_tracker(g; verbosity = 1, print_interval = 2)
    t0 = time()
    i = 0
    fmax = Inf
    xmax = nothing
    (f, dparams, params) -> begin
        if i == 0
            println(" eval   | current    | best")
            println("_"^33)
        end
        i += 1
        ret = g(f, dparams, params)
        if ret !== nothing && ret < fmax
            fmax = ret
            xmax = copy(params)
        end
        t1 = time()
        if verbosity > 0 && t1 - t0 > print_interval
            GC.gc()
            t0 = t1
            @printf "%7i | %10.6g | %10.6g\n" i -ret -fmax
        end
        ret
    end
end
function combine_gradient_funcs(gs)
    length(gs) == 1 && return gs[]
    (f, dparams, params) -> begin
        zero!(dparams)
        d = [zero(dparams) for _ in 1:Threads.nthreads()]
        dlogp = 0.
        lk = ReentrantLock()
        Threads.@threads for g in gs
            ret = g(f, d[Threads.threadid()], params)
            lock(lk)
            try
                dlogp += ret
                dparams .+= d[Threads.threadid()]
            finally
                unlock(lk)
            end
        end
        dparams ./= length(gs)
        f !== nothing && return dlogp / length(gs)
    end
end
function split_track(track, n = min(1, Threads.nthreads()÷2))
    n == 1 && return track
    T = nrow(track)
    L = T ÷ n
    [track[i*L+1:(i+1)*L, :] for i in 0:n-1]
end
"""
$SIGNATURES

Train a model to a single data frame `data` with initial parameters `params` (=`params(model)` by default).
"""
function train(model, data::DataFrame, params = FlyRL.params(model);
        multi_threading = false, nthreads = Threads.nthreads() ÷ 2, kwargs...)
    if multi_threading
        train(model, split_track(data, nthreads), params; kwargs...)
    else
        train(model.agent, [preprocess(model.preprocessor, data)], params; kwargs...)
    end
end
"""
$SIGNATURES

Train a model to a multiple data frames `data` with initial parameters `params` (=`params(model)` by default).
"""
function train(model, data::AbstractVector, params = FlyRL.params(model); kwargs...)
    train(model.agent, [preprocess(model.preprocessor, d) for d in data], params; kwargs...)
end
"""
$SIGNATURES

Train agent on preprocessed `data` with initial parameters `params` (=`params(model)` by default).

Keep the default `opt = :MLSL` for small scale problems and set `opt = Adam(), batchsize = 32`,
if training on minibatches of size 32 is desired. Fix parameter values with e.g. `fixed = (; η = 0.)`.
Gradient evaluation is multi-threaded, if julia is started with multiple threads, e.g. `bash> julia -t8`.
"""
function train(
    agent,
    data::AbstractVector{<:NamedTuple{(:input, :target, :shock)}},
    params = FlyRL.params(agent);
    epochs = 10^6,
    verbosity = 1,
    opt = :MLSL,
    print_interval = 2,
    lopt = :LD_SLSQP,
    maxtime = 60,
    optim_options = Optim.Options(iterations = epochs, time_limit = maxtime),
    maxeval = epochs,
    lb = -100,
    ub = 100,
    fixed = nothing,
    batchsize = nothing
)
    params = remove_fixed(params, fixed)
    dparams = zero(params)
    ax = getaxes(params)
    logp = 0.0
    gs = [gradient_func([0.0], [1.0], deepcopy(agent), zero(agent),
                        d.input, zero(d.input),
                        d.target, zero(d.target),
                        d.shock, zero(d.shock), ax, fixed; batchsize)
          for d in data]
    g! = wrap_tracker(combine_gradient_funcs(gs); verbosity, print_interval)
    if isa(opt, AbstractOptimiser)
        tstart = time()
        for _ in 1:epochs
            logp = g!(true, dparams, params)
            params .-= apply!(opt, params, dparams)
            time() - tstart > maxtime && break
        end
        extra = nothing
    elseif opt == :MLSL
        o = Opt(:G_MLSL_LDS, length(params))
        o.lower_bounds = lb
        o.upper_bounds = ub
        o.local_optimizer = Opt(lopt, length(params))
        o.min_objective = (params, dparams) -> g!(true, dparams, params)
        o.maxtime = maxtime
        o.maxeval = maxeval
        logp, xsol, extra = NLopt.optimize(o, params)
        params .= xsol
    else # Optim
        extra = Optim.optimize(Optim.only_fg!(g!), params, opt, optim_options)
        logp = Optim.minimum(extra)
        params .= Optim.minimizer(extra)
    end
    if batchsize === nothing && g!.fmax.contents < logp
        logp = g!.fmax.contents
        params .= g!.xmax.contents
    end
    params = append_fixed(params, fixed)
    (; logp = -logp, params, extra)
end
