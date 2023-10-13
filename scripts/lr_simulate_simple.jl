using Pkg, Distributed
Pkg.activate(@__DIR__)
@everywhere begin
using FlyRL, DataFrames, Serialization, FiniteDiff, Random, ComponentArrays
import FlyRL: Preprocessor, DynamicCompressEncoder, VectorEncoder, ColumnPicker,
              SemanticEncoder7, ShockArmEncoder, ArmEncoder, MarkovKEncoder,
              OrientationEncoder, VelocityEncoder, DeltaPositionEncoder,
              AngleEncoder, SpeedEncoder, FourWallsEncoder, decode,
              InShockArmEncoder, simulate, LevelEncoder, FilterEncoder,
              Model, PolicyGradientAgent, params, logprob, preprocess, logprob!,
              plot_track, plot_maze, plot_probs, Environment, train, plot_compare_probs
preprocessor = Preprocessor(input = ShockArmEncoder() |>
                                    x -> FilterEncoder(d -> d.state .!= "center", x) |>
                                    x -> DynamicCompressEncoder(:state, x, max_steps = 150) |>
                                    x -> VectorEncoder(x, intercept = true)
                                    ,
                                    target = ShockArmEncoder() |>
                                             LevelEncoder);
alt_preprocessor = Preprocessor(input = VectorEncoder(ShockArmEncoder(), intercept = true),
                                target = LevelEncoder(ShockArmEncoder()))
model = Model(PolicyGradientAgent(Din = 5, Dout = 3), preprocessor);
alt_model = Model(PolicyGradientAgent(Din = 5, Dout = 3), alt_preprocessor);
function in_shock_arm(x::ComponentVector)
    x.shock == 1 && return true
    hasproperty(x, :shockˌ1) && x.shockˌ1 == 1 && return true
    false
end
in_shock_arm(x, y) = false
env = Environment(; preprocessor = alt_preprocessor, shock = in_shock_arm);
include("helper.jl")
end

sim_results = vcat([[joinpath(root, f) for f in fs if match(r"fit2-.*.dat", f) !== nothing]
                    for (root, _, fs) in walkdir("../data/")]...)

@sync @distributed for sim_result_fn in sim_results
    sim_result = deserialize(sim_result_fn)
    fn = sim_result.filename
    tmp = splitpath(fn)
    tmp[end] = "simresult2-$(tmp[end][1:end-4]).dat"
    fitfn = joinpath(tmp)
    isfile(fitfn) && continue
    @info "starting simulation $fn"
    θ = sim_result.θ
    results = []
    for seed in 1:32
        FlyRL.rand_state!(env, rng = Xoshiro(seed))
        x, s, = simulate(alt_model.agent, env, θ, sim_result.n_decisions + 1,
                         rng = Xoshiro(seed))
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
    serialize(fitfn, results)
end
