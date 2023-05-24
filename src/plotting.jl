function _color(track)
    if hasproperty(track, :pattern)
        String.(encode(ColorEncoder(with_outliers = true), track).color)
    else
        "blue"
    end
end
"""
$SIGNATURES

Plot the outline of a maze and the positions of a fly (`track.x` and `track.y`).
See also [`plot_track`](@ref). `f = Makie.Figure()`.
"""
function plot_maze(track; f = Figure())
    ax = Makie.Axis(f[1, 1], aspect = 200 / 180, yreversed = true)
    grid = vec([(x, y) for x ∈ 0:.2:200, y ∈ 0:.2:180])
    heatmap!(ax,
        first.(grid),
        last.(grid),
        in_maze.(first.(grid), last.(grid)),
        colormap = :grays,
        colorrange = (-1, 1),
    )
    scatter!(
        track.x,
        track.y,
        strokecolor = ifelse.(hasproperty(track, :shock) && track.shock, :yellow, :white),
#         markersize = ifelse.(in_maze.(track.x, track.y), 0, 6),
        markersize = 6,
        strokewidth = ifelse.(hasproperty(track, :shock) && track.shock, 1, 0),
#         strokewidth = 0,
        color = _color(track),
    )
    hidedecorations!(ax)
    f
end

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

Plots a track as a function of time (see [`plot_track_3D`](@ref)) and on the maze outline (see [`plot_maze`](@ref)).
"""
function plot_track(track; f = Figure(resolution = (940, 320)))
#     Box(f[1, 1], color = (:red, 0.2), strokewidth = 0)
#     Box(f[1, 2], color = (:green, 0.2), strokewidth = 0)
    plot_track_3D(track, f = f[1, 1])
    plot_maze(track, f = f[1, 2])
    colsize!(f.layout, 1, Auto(1))
    colsize!(f.layout, 2, Auto(.6))
    f
end
const DEFAULT_Δt = .15
"""
$SIGNATURES

Plots `track.x` and `track.y` as a function of `track.t`.
"""
function plot_track_3D(track;
        f = Figure(), cm_per_pixel = 2.5/200,
        ylabel = "x [cm]",
        xlabel = "time [s]",
        zlabel = "y [cm]",
        perspectiveness = 0.3,
        elevation = 0.15,
        azimuth = -1.1,
        viewmode = :stretch,
        alignmode = Outside(30),
        protrusions = (-30, 0, 0, -80),
        yticks = Makie.WilkinsonTicks(4, k_max = 5),
        zticks = Makie.WilkinsonTicks(4, k_max = 5),
    )
    ax = Axis3(f[1, 1]; ylabel, xlabel, zlabel, perspectiveness, elevation,
                        azimuth, viewmode, alignmode, protrusions, yticks, zticks)
    if hasproperty(track, :t)
        time = track.t
    else
        time = (1:length(track.x)) * DEFAULT_Δt
    end
    if hasproperty(track, :shock)
        scatter!(
            ax,
            time[track.shock],
            track.x[track.shock] * cm_per_pixel,
            (180 .- track.y[track.shock]) * cm_per_pixel,
            color = :yellow,
            markersize = 8,
        )
    end
    lines!(
        ax,
        time,
        track.x * cm_per_pixel,
        (180 .- track.y) * cm_per_pixel,
        color = _color(track),
    )
    ylims!(ax, (0, 180*cm_per_pixel))
    zlims!(ax, (0, 170*cm_per_pixel))
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
