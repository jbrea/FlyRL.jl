using Pkg, Distributed
Pkg.activate(@__DIR__)
@everywhere begin
using FlyRL, DataFrames, Serialization, FiniteDiff, Random
import FlyRL: Preprocessor, DynamicCompressEncoder, VectorEncoder, ColumnPicker,
              SemanticEncoder7, ShockArmEncoder, ArmEncoder, MarkovKEncoder,
              OrientationEncoder, VelocityEncoder, DeltaPositionEncoder,
              AngleEncoder, SpeedEncoder, FourWallsEncoder, decode,
              InShockArmEncoder, simulate, LevelEncoder, FilterEncoder,
              Model, PolicyGradientAgent, params, logprob, preprocess, logprob!,
              plot_track, plot_maze, plot_probs, Environment, train, plot_compare_probs
preprocessor = Preprocessor(input = ShockArmEncoder() |>
                                    x -> FilterEncoder(d -> d.shock_arm .!= "center", x) |>
                                    x -> DynamicCompressEncoder(:shock_arm, x, max_steps = 150) |>
                                    x -> VectorEncoder(x, intercept = true)
                                    ,
                                    target = ShockArmEncoder() |>
                                             LevelEncoder);
alt_preprocessor = Preprocessor(input = VectorEncoder(ShockArmEncoder(), intercept = true),
                                target = LevelEncoder(ShockArmEncoder()))
model = Model(PolicyGradientAgent(Din = 5, Dout = 3), preprocessor);
alt_model = Model(PolicyGradientAgent(Din = 5, Dout = 3), alt_preprocessor);
env = Environment(; preprocessor, shock = FlyRL.in_shock_arm);
function lr_scan(l0, model, input, target, shock, params, dp)
    for η in 2. .^ (-60:0)
        l1 = logprob(model, input, target, shock, params + η * dp)
        if l1 < l0
#             @show η/2
            return η/2
        end
        η *= 2
    end
    0. # no fine tuning
end
function gradient_descent_fine_tuning(model, track, params0; T = 10^3)
    params = copy(params0)
    input, target, shock = preprocess(model.preprocessor, track)
    l0, dp0 = FlyRL.grad_logprob(model, input, target, shock, params)
    l00 = l0
    η = lr_scan(l0, model, input, target, shock, params, dp0)
    if η > 0
        for _ in 1:T
            params .+= η * dp0
            l1, dp1 = FlyRL.grad_logprob(model, input, target, shock, params)
            if l1 < l0
                params .-= η * dp0
                η = lr_scan(l0, model, input, target, shock, params, dp0)
                η == 0 && break
            else
                dp0 = dp1
                l0 = l1
            end
        end
    end
    (; params, l0, dl = l0 - l00, dparams = params - params0,
       maxdparams = maximum(abs, params - params0),
       dp = dp0, maxdp = maximum(abs, dp0))
end
function laplace_approx(model, track, θ)
    H = FiniteDiff.finite_difference_hessian(θ -> logprob(model, track, θ), θ)
    H[2, 2] ≈ 0 && return Inf
    -1/H[2, 2]
end
end

sim_results = vcat([[joinpath(root, f) for f in fs if match(r"fit-.*.dat", f) !== nothing]
                    for (root, _, fs) in walkdir("../data/")]...)

@sync @distributed for sim_result_fn in sim_results
    sim_result = deserialize(sim_result_fn)
    fn = sim_result.filename
    @info "starting simulation $fn"
    θ = sim_result.θ
    results = []
    for seed in 1:32
        Random.seed!(seed)
        x, s, = simulate(model.agent, env, θ, sim_result.n_decisions + 1)
        track = decode(model.preprocessor.input, x[2:end]) |> DataFrame
        track.shock = s
        res = train(alt_model, track, θ, maxtime = 60, print_interval = 10)
        gd_res = gradient_descent_fine_tuning(alt_model, track, res.params)
        lp = laplace_approx(alt_model, track, gd_res.params)
        input, target, shock = preprocess(alt_preprocessor, track)
        dp = FlyRL.grad_logprob(alt_model, input, target, shock, gd_res.params)[2]
        l = length(target)
        result = (seed = seed, track = track, θ = gd_res.params, θ0 = res.params,
                  η = gd_res.params.η, sigma_η = lp,
                  n_decisions = l, filename = fn, dp = dp, dp_norm = maximum(abs, dp))
        push!(results, result)
    end
    tmp = splitpath(fn)
    tmp[end] = "simresult-$(tmp[end][1:end-4]).dat"
    fitfn = joinpath(tmp)
    serialize(fitfn, results)
end
