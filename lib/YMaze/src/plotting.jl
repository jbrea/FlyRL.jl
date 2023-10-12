function _color(track)
    if hasproperty(track, :pattern)
        color.(track.pattern, track.x, track.y)
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

