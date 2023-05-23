"""
    Model(agent, preprocessor)

Makes a model.
"""
struct Model{A, P}
    agent::A
    preprocessor::P
end
state(model::Model) = state(model.agent)
params(model::Model) = params(model.agent)

"""
$SIGNATURES

Compute the log-probability of the `data` given the `model` with parameters `params`.
"""
function logprob(model::Model, data::DataFrame, params)
    input, target, shock = preprocess(model.preprocessor, data)
    logprob(model, input, target, shock, params)
end

function logprob(model::Model, input, target, shock, params)
    res = [0.]
    logprob!(res, model.agent, input, target, shock, params)
    res[]
end
function logprob!(res, agent, input, target, shock, params)
    res[] = 0
    for i in eachindex(target)
        res[] += logprob(agent, input[i], target[i], shock[i], params)
    end
    nothing
end

function softmax(logits)
    m = maximum(logits)
    em = exp.(logits .- m)
    em ./ sum(em)
end
function softmax!(π, model, x)
    rm = 0.
    for i in eachindex(π)
        π[i] = tmp = model(i, x)
        if tmp > rm
            rm = tmp
        end
    end
    s = 0.
    for i in eachindex(π)
        π[i] = exp(π[i] - rm)
        s += π[i]
    end
    for i in eachindex(π)
        π[i] /= s
    end
    π
end

struct CircularArray{T,L} <: AbstractVector{T}
    x::Vector{T}
end
CircularArray(N, L; T = Float64) = CircularArray{Vector{T}, L}([zeros(T, N) for _ in 1:L])
function Base.getindex(x::CircularArray{T,L}, i) where {T,L}
    idx = (i-1)%L + 1
    x.x[idx]
end
function Base.setindex!(x::CircularArray{T,L}, v, i) where {T,L}
    idx = (i-1)%L + 1
    setindex!(x.x, v, idx)
end
Base.size(::CircularArray{T, L}) where {T, L} = (L,)
Base.zero(x::CircularArray{T,L}) where {T, L} = CircularArray{T, L}(zero(x.x))

struct LinearModel{T}
    w::Matrix{T}
end
LinearModel(; Din, Dout, T = Float64) = LinearModel{T}(zeros(T, Dout, Din))
function (m::LinearModel{T})(i, x) where T
    res = zero(T)
    w = m.w
    i > size(w, 1) && return res
    for j in eachindex(x)
        res += w[i, j] * x[j]
    end
    res
end
zero!(m::LinearModel) = m.w .= 0
Base.zero(m::LinearModel) = LinearModel(zero(m.w))
initialize!(m::LinearModel, params) = m.w .= params.w
params(m::LinearModel) = (; w = zero(m.w))
function update!(p::LinearModel, input, π, shock, t, L, params)
    t ≤ L && return p
    γ = sigmoid(params.logitγ)
    if γ > 0
        G = future_discounted_shocks(γ, shock, t, L)
    else
        G = shock[t-L]
    end
    if G != 0
        η = params.η * G
        for i in axes(p.w, 1), j in axes(p.w, 2)
            # π is here (π - target)
            p.w[i, j] += η * π[t-L][i] * input[t-L][j]
        end
    end
    p
end

struct DeltaPositionModel{E,F,S,SE}
    dx::Vector{Int}
    dy::Vector{Int}
    state::S
    encoded::SE
    enc::E
    p0::Base.RefValue{Float64}
    f::F
end
Base.zero(m::DeltaPositionModel) = DeltaPositionModel(copy(m.dx), copy(m.dy), zero(m.state),
                                                        zero(m.encoded), deepcopy(m.enc), Ref(0.), zero(m.f))
function zero!(m::DeltaPositionModel)
    zero!(m.state)
    zero!(m.encoded)
    m.p0[] = 0
    zero!(m.f)
    m
end
function initialize!(m::DeltaPositionModel, params)
    m.p0[] = params.p0
    initialize!(m.f, params)
end
params(m::DeltaPositionModel) = ComponentArray(merge((; p0 = 0.), params(m.f)))
function delta_position_grid(; maxradius = 20, stepsize = 1)
    d = [(x, y) for x in -maxradius:stepsize:maxradius,
                    y in -maxradius:stepsize:maxradius
                    if x^2 + y^2 ≤ maxradius^2]
    first.(d), last.(d)
end
"""
$SIGNATURES

Make a DeltaPositionModel.
"""
function DeltaPositionModel(; nhidden = 32, σ = tanh, T = Float64,
        dx = nothing, dy = nothing, f = nothing,
        encoder = VectorEncoder(EightWallsEncoder(),
                                MarkovKEncoder(2, AngleEncoder2D()),
                                MarkovKEncoder(2, SpeedEncoder())
                               ),
        kwargs...)
    if dx === nothing && dy === nothing
        dx, dy = delta_position_grid(; kwargs...)
    end
    state = ComponentArray(x = zero(T), y = zero(T),
                           Δx = zero(T), Δy = zero(T),
                           Δt = zero(T),
                           ox = zero(T), oy = zero(T),
                           oxˌ1 = zero(T), oyˌ1 = zero(T),
                           speed = zero(T),
                           speedˌ1 = zero(T),
                           angle = zero(T),
                           angleˌ1 = zero(T)
                          )
    ks = encoder.keys
    encoded = ComponentArray(NamedTuple{ks}(zeros(T, length(ks))))
    if f === nothing
        f = MLP(DenseLayer(length(encoded), nhidden, σ; T),
                DenseLayer(nhidden, 1, identity; T))
    end
    DeltaPositionModel(dx, dy, state, encoded, encoder, Ref(0.), f)
end
EnzymeRules.inactive(::typeof(encode!), args...) = nothing
@inline function (d::DeltaPositionModel)(i, x)
    if i == 1 # needs only be done once per step
        d.state.oxˌ1 = x.oxˌ1
        d.state.oyˌ1 = x.oyˌ1
        d.state.speedˌ1 = x.speedˌ1
        d.state.angleˌ1 = x.angleˌ1
        d.state.Δt = x.futureΔt
    end
    dx = d.dx[i]
    dy = d.dy[i]
#     show = i == Main.target[Main.t]
#     if show
#         @show Main.t (dx, dy)
#     end
    dx == dy == 0 && return d.p0[]
    d.state.x = x.x + dx
    d.state.y = x.y + dy
#     if show
#         in_maze(d.state.x, d.state.y)
#     end
    in_maze(d.state.x, d.state.y) || return -Inf
    d.state.Δx = dx
    d.state.Δy = dy
    encode!(d.enc, d.encoded, d.state)
    ret = d.f(d.encoded)[]
#     if show
#         @show d.encoded d.state ret
#     end
    ret
end
struct PolicyGradientAgent{T,M,L}
    model::M
    π::CircularArray{Vector{T},L}
end
"""
$SIGNATURES

"""
function PolicyGradientAgent(; Din, Dout, T = Float64,
        model = LinearModel(; Din, Dout = Dout-1, T), update_lag = 5)
    PolicyGradientAgent{T, typeof(model), update_lag+1}(model,
                                         CircularArray(Dout, update_lag+1; T))
end
update!(::DeltaPositionModel, input, π, shock, t, L, params) = nothing
function wsample(π)
    θ = rand()
    s = 0.
    for i in eachindex(π)
        s += π[i]
        s > θ && return i
    end
    return length(π)
end
function _findfirst(x::ComponentVector)
    for i in eachindex(x)
        x[i] == 1 && return i
    end
end
function simulate(agent::PolicyGradientAgent{T,M,L}, env, params, N) where {T,M,L}
    initialize!(agent, params)
    states = [copy(state(env))]
    shocks = Float64[]
    actions = Int[]
    π = agent.π
    logprob = zero(T)
    for t in 1:N
        softmax!(π[t], state(agent), states[end])
#         softmax!(π[t], state(agent), Main.input[t])
#         @show t sum(state(agent)
        a = wsample(π[t])
#         a = _findfirst(Main.target[t])
        logprob += log(π[t][a])
        π[t][a] -= 1
        s′, shock = step!(env, a)
#         shock = Main.shock[t]
        push!(shocks, shock)
        push!(actions, a)
        update!(agent, states, π, shocks, t, L-1, params)
        push!(states, copy(s′))
    end
    (; states, shocks, actions, logprob = logprob/N)
end
function Base.zero(p::PolicyGradientAgent{T,M,L}) where {T,M,L}
    PolicyGradientAgent{T,M,L}(zero(p.model), zero(p.π))
end
zero!(p::PolicyGradientAgent) = zero!(p.model)
params(p::PolicyGradientAgent) = ComponentArray(; merge((logitγ = 0., η = 0.), params(p.model))...)
sigmoid(x) = 1/(1 + exp(-x))
target(t::AbstractVector, i) = t[i]
target(t::Int, i) = t == i
function update!(p::PolicyGradientAgent, input, π, shock, t, L, params)
    update!(p.model, input, π, shock, t, L, params)
end
state(p::PolicyGradientAgent) = p.model
function initialize!(p::PolicyGradientAgent, params)
    initialize!(p.model, params)
    p
end
function future_discounted_shocks(γ, shocks, t, L)
    G = 0.
    γeff = 1.
    for k in 1:L
        G += γeff * shocks[t - L + k]
        γeff *= γ
    end
    G
end
_minus_target!(π, target::Int) = π[target] -= 1
_minus_target!(π, target::AbstractVector{<:Bool}) = π .-= target
function logprob!(res, agent::PolicyGradientAgent{T,M,L}, input, target, shock, params) where {T,M,L}
    initialize!(agent, params)
    res[] = zero(T)
    π = agent.π
    for t in eachindex(target)
        softmax!(π[t], state(agent), input[t])
#         @show t π[t][target[t]] target[t] maximum(π[t]) minimum(π[t])
        res[] += log(π[t][target[t]])
        _minus_target!(π[t], target[t]) # modifies π to (π - target)
        update!(agent, input, π, shock, t, L-1, params)
    end
    res[] /= length(target)
    nothing
end

struct StationaryAgent{M,T}
    model::M
    π::Vector{T}
end
params(s::StationaryAgent) = params(s.model)
"""
$SIGNATURES
"""
StationaryAgent(; Dout, model, T = Float64) = StationaryAgent(model, zeros(T, Dout))
Base.zero(s::StationaryAgent) = StationaryAgent(zero(s.model), zero(s.π))
function zero!(s::StationaryAgent)
    zero!(s.model)
    zero!(s.π)
end
function logprob!(res, agent::StationaryAgent{M,T}, input, target, shock, params) where {M, T}
    initialize!(agent.model, params)
    π = agent.π
    res[] = zero(T)
    for t in eachindex(target)
        softmax!(π, agent.model, input[t])
#         @show agent.model.state agent.model.encoded input[t]
#         Main.t += 1
#         @show maximum(π) sort(π)[end-10:end] π[target[t]]
        res[] += log(π[target[t]])
#         @show res[]
    end
    res[] /= length(target)
    nothing
end
function simulate(agent::StationaryAgent{M,T}, env, params, N) where {M,T}
    initialize!(agent.model, params)
    states = [copy(state(env))]
    shocks = Float64[]
    actions = Int[]
    π = agent.π
    logprob = zero(T)
    for t in 1:N
        softmax!(π, agent.model, states[end])
#         @show agent.model.state agent.model.encoded states[end]
        a = wsample(π)
#         a = Main.target[t]
        logprob += log(π[a])
#         @show logprob
        s′, shock = step!(env, a)
#         @show s′
        push!(shocks, shock)
        push!(actions, a)
        push!(states, copy(s′))
    end
    (; states, shocks, actions, logprob = logprob/N)
end
