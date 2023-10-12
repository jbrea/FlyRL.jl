cartesian(r, ϕ, trig) = ifelse(r == 0, zero(r), r * trig(ϕ + π / 2))
function plot_velocity(track; egocentric = true,
                              heatmap = true,
                              colormap = :bilbao,
                              f = Figure())
    ax = Makie.Axis(f[1, 1], autolimitaspect = 1)
    if egocentric
        vx = cartesian.(track.speed, track.angles, cos)
        vy = cartesian.(track.speed, track.angles, sin)
    else
        vx = track.vx
        vy = track.vy
    end
    if heatmap
        h = fit(
            Histogram,
            (collect(skipmissing(vx)), collect(skipmissing(vy))),
            (-29.5:29.5, -29.5:29.5),
        )
        CairoMakie.heatmap!(ax,
            h.edges[1],
            h.edges[2],
            log.(h.weights .+ 1),
            colormap = colormap,
        )
    else
        scatter!(ax, vx, vy)
    end
    f
end

"""
$SIGNATURES

Plots the probabilities of changes in x and y direction for a [`DeltaPositionModel`](@ref).
"""
function plot_delta_pos_probs(agent; transform = log, kwargs...)
    plot_delta_pos_probs(transform.(agent.π),
                         agent.model; kwargs...)
end
"""
$SIGNATURES
"""
function plot_delta_pos_probs(π, delta_position_model; f = Figure(resolution = (400, 400)), colormap = :bilbao)
    ax = Makie.Axis(f[1, 1], autolimitaspect = 1, yreversed = true, ylabel = "Δy", xlabel = "Δx")
    CairoMakie.heatmap!(ax,
                        delta_position_model.dx,
                        delta_position_model.dy,
                        π,
                        colormap = colormap)
    ox = [delta_position_model.state.oxˌ1]
    oy = [delta_position_model.state.oyˌ1]
    s = delta_position_model.state.speedˌ1 * DEFAULT_Δt
    arrows!(ax, -s*ox, -s*oy, s*ox, s*oy, color = :white, linewidth = 3)
    f
end

_level(v::CategoricalValue) = levelcode(v)
_levels(v::CategoricalValue) = CategoricalArrays.levels(v)
_level(v::AbstractVector{<:Number}) = findfirst(==(1), v)
_levels(v::AbstractVector{<:Number}) = eachindex(v)
function _level(v::AbstractVector{<:CategoricalValue})
    b = 1
    res = 1
    for vi in v
        res += b * (levelcode(vi)-1)
        b *= length(_levels(vi))
    end
    res
end
_levels(v::AbstractVector{<:CategoricalValue}) = collect(Iterators.product(CategoricalArrays.levels.(v)...))
function transition_probabilities(x, x′)
    m = zeros(length(_levels(x′[1])), length(_levels(x[1])))
    for i in eachindex(x′)
        m[_level(x′[i]), _level(x[i])] += 1
    end
    for j in axes(m, 2)
        sm = sum(m[i, j] for i in axes(m, 1))
        if sm > 0
            m[:, j] ./= sm
        end
    end
    m
end
function plot_transition_probs(track::DataFrame;
        preprocessor = Preprocessor(input = DynamicCompressEncoder(ShockArmEncoder()),
                                    target = ShockArmEncoder()))
    input, target, = preprocess(preprocessor, track)
    m = transition_probabilities(input, target)
    ls = vec(join.(_levels(input[1]), ""))
    ls′ = vec(join.(_levels(target[1]), ""))
    plot_transition_probs(m, ls, ls′)
end
function plot_transition_probs(m, ls, ls′)
    heatmap(m', axis = (autolimitaspect = 1,
                       ylabel = "to", xlabel = "from",
                       xticklabelrotation = π/4,
                       xticks = (eachindex(ls), ls),
                       yticks = (eachindex(ls′), ls′)))
end
function probabilities(x)
    m = zeros(length(_levels(x[1])))
    for i in eachindex(x)
        m[_level(x[i])] += 1
    end
    m ./ length(x)
end
"""
$SIGNATURES

Plot probabilities (frequencies) of being in certain states.
By default the states defined by the `ShockArmEncoder` are taken, i.e.
`preprocessor = Preprocessor(input = DynamicCompressEncoder(:shock_arm, ShockArmEncoder()),
                             target = ShockArmEncoder())`
"""
function plot_probs(track::DataFrame;
        preprocessor = Preprocessor(input = DynamicCompressEncoder(:shock_arm, ShockArmEncoder()),
                                    target = ShockArmEncoder()), kwargs...)
    input, = preprocess(preprocessor, track)
    plot_probs(input; kwargs...)
end
function _compute_probs(input; range = (0, 1))
    n = length(input)
    l = floor(Int, range[1]*n) + 1
    u = ceil(Int, range[2]*n)
    probabilities(input[l:u])
end
function plot_probs(input; range = (0, 1), kwargs...)
    if isa(first(input), AbstractVector)
        ms = _compute_probs.(input; range)
        m = sum(ms) ./ length(ms)
        v1 = input[1][1]
    else
        m = _compute_probs(input; range)
        v1 = input[1]
    end
    plot_probs(m, vec(join.(_levels(v1), "")); kwargs...)
end
function plot_probs(m, ls; f = Figure(), kwargs...)
    ax = Makie.Axis(f[1, 1];
                    xticks = (eachindex(ls), ls),
                    xticklabelrotation = π/4,
                    kwargs...)
    barplot!(ax, m)
    f
end
"""
$SIGNATURES

Compare the probabilities of `track1` and `track2` for being in certain states
at different temporal ranges. The default ranges are `(0, 1)` (full range),
`(0, .5)` (first half) `(.5, 1))` (second half).
`kwargs` are passed to [`plot_probs`](@ref).
"""
function plot_compare_probs(track1, track2; ranges = ((0, 1), (0, .5), (.5, 1)),
                                            f = Figure(), kwargs...)
    for (i, t) in pairs((track1, track2)), (j, r) in pairs(ranges)
        plot_probs(t; range = r, f = f[i, j], title = string(r), kwargs...)
    end
    f
end

"""
$SIGNATURES

"""
function plot_summary(summary, tracks; f = Figure(), sortperm = 1:length(tracks))
    data = summarize.(Ref(summary), tracks)
    ax = Makie.Axis(f[1, 1], xlabel = "track id", ylabel = replace(string(keys(data[1])[1]), '_' => ' '))
    scatter!(ax, eachindex(data), first.(data)[sortperm])
    hlines!(ax, chance_level(summary))
    f
end
"""
$SIGNATURES

Plot some summary statistics for a collection of `tracks`.
"""
function plot_summaries(tracks; f = Figure(),
        summaries = (RelativeTimeInShockArm(), ChangeOf(RelativeTimeInShockArm()),
                     RelativeVisitsToShockArm(), ChangeOf(RelativeVisitsToShockArm())),
        sortperm = 1:length(tracks))
    for (i, summary) in pairs(summaries)
        plot_summary(summary, tracks; f = f[1, i], sortperm)
    end
    f
end
