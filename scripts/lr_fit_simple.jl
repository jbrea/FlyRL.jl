using Pkg, Distributed
Pkg.activate(@__DIR__)
@everywhere begin
using FlyRL, DataFrames, Serialization, FiniteDiff
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
model = Model(PolicyGradientAgent(Din = 5, Dout = 3), preprocessor);
function laplace_approx(model, track, θ)
    H = FiniteDiff.finite_difference_hessian(θ -> logprob(model, track, θ), θ)
    H[2, 2] ≈ 0 && return Inf
    -1/H[2, 2]
end
include("helper.jl")
end

# @time tracks = FlyRL.read_directory("../data/",
#                                     drop_outliers = true,
#                                     pattern = r"^track",
#                                     warn_outliers = false);
tracks = deserialize("../data/tracks.dat")
tracks = filter(x -> !any(x.raw_shock), tracks);
# pretend they the get shocked
for track in tracks
    nrow(track) > 0 || continue
    if any(track.raw_shock)
        track.shock = track.raw_shock
    else
        track.shock = track.state .== "neutral left"
    end
end

@sync @distributed for track in tracks
    nrow(track) > 500 || continue
    fn = metadata(track)["filename"]
    @info "starting fit $fn"
    θ = params(model)
    res = train(model, track, θ, maxtime = 60, print_interval = 10)
    gd_res = gradient_descent_fine_tuning(model, track, res.params)
    lp = laplace_approx(model, track, gd_res.params)
    input, target, shock = preprocess(preprocessor, track)
    dp = FlyRL.grad_logprob(model, input, target, shock, gd_res.params)[2]
    l = length(target)
    result = (θ = gd_res.params, θ0 = res.params,
              η = gd_res.params.η, sigma_η = lp,
              n_decisions = l, filename = fn, dp = dp, dp_norm = maximum(abs, dp))
    tmp = splitpath(fn)
    tmp[end] = "fit2C-$(tmp[end][1:end-4]).dat"
    fitfn = joinpath(tmp)
    serialize(fitfn, result)
end
