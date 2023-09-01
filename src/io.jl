"""
$SIGNATURES

Read track `f` in directory `root` and corresponding time, shock and pattern file.
Returns a `DataFrame`. `kwargs` are passed to [`preprocess`](@ref).
"""
function read(root, f; preprocess = true, kwargs...)
    track = CSV.read(joinpath(root, f), DataFrame, header = [:x, :y])
    t = CSV.read(joinpath(root, replace(f, "track" => "time")), DataFrame, header = [:time])
    track.t = (t.time .- t.time[1]) ./ 1e9 # time in s
    s = CSV.read(
        joinpath(root, replace(f, "track" => "shock")),
        DataFrame,
        header = [:shock],
    )
    track.shock = s.shock .== "X"
    p = CSV.read(
        joinpath(root, replace(f, "track" => "pattern")),
        DataFrame,
        header = [:pattern],
    )
    track.pattern = p.pattern
    if preprocess
        track = FlyRL.preprocess(track; kwargs...)
    end
    metadata!(track, "filename", joinpath(root, f), style = :note)
    track
end
read(f; kwargs...) = read(splitdir(f)...; kwargs...)

"""
$SIGNATURES

Read all tracks in directory `dir`.
Uses [`read`](@ref) and passes `kwargs` to [`preprocess`](@ref).
"""
function read_directory(dir; verbosity = 1, pattern = r"^track", kwargs...)
    tracks = DataFrame[]
    for (root, _, files) ∈ walkdir(dir)
        for f ∈ files
            match(pattern, f) !== nothing || continue
            if verbosity > 0
                println("Reading file $(joinpath(root, f)).")
            end
            push!(tracks, read(root, f; kwargs...))
        end
    end
    tracks
end

function drop_initial_outliers(df; starttime = 30, tolerance = 2)
    starti = findfirst(>(starttime), df.t)
    tolerance = tolerance
    for (i, r) ∈ Iterators.drop(pairs(eachrow(df)), starti)
        if !in_maze(r.x, r.y)
            tolerance = tolerance
        elseif tolerance == 0
            starti = i
            break
        else
            tolerance -= 1
        end
    end
    df[starti:end, :]
end

drop_outliers(df) = df[in_maze.(df.x, df.y), :]

function roll(df; filter = rollmedian, window = 5)
    DataFrame(
        x = Int.(filter(df.x, window)),
        y = Int.(filter(df.y, window)),
        t = filter(df.t, window),
    )
end

function subsample(track; resolution = 5)
    df = copy(track)
    df.x = round.(track.x / resolution) * resolution
    df.y = round.(track.y / resolution) * resolution
    df
end

function detect_outliers(df)
    for (i, r) ∈ pairs(eachrow(df))
        if !in_maze(r.x, r.y)
            @warn "Detected outlier in row $i: $((r.x, r.y))"
        end
    end
end

function calibrate_simple!(df)
    Δx = 0
    Δy = 0
    best = -1000
    for dx in -20:20
        for dy in -20:20
            n_inmaze = sum(in_maze.(df.x .+ dx, df.y .+ dy))
            if n_inmaze > best
                best = n_inmaze
                Δx = dx
                Δy = dy
            end
        end
    end
    df.x .+= Δx
    df.y .+= Δy
    df
end
function offset_to_closest(x, y)
    closest = Inf
    dx = 0.
    dy = 0.
    for (xr, yr) in CENTER_POINTS
        d = (x - xr)^2 + (y - yr)^2
        if d < closest
            closest = d
            dx = xr - x
            dy = yr - y
        end
    end
    dx, dy
end
function calibrate!(df)
    offsets = offset_to_closest.(df.x, df.y)
    dx = round(Int, median(first.(offsets)))
    dy = round(Int, median(last.(offsets)))
    df.x .+= dx
    df.y .+= dy
    abs(dx) + abs(dy)
end

"""
$SIGNATURES

Preprocess data frame `df`.
"""
function preprocess(
    df;
    filter = rollmedian,
    window = 1,
    initial_outlier_tolerance = 2,
    encoders = (VelocityEncoder(), DeltaTimeEncoder(),
                SpeedEncoder(), OrientationEncoder(), AngleEncoder()),
    warn_outliers = false,
    drop_outliers = true,
    subsample_resolution = 1,
    calibrate = true
)
    df = drop_initial_outliers(df, tolerance = initial_outlier_tolerance)
    if calibrate
        for i in 1:6
            Δ = calibrate!(df)
            Δ == 0 && break
        end
    end
    if warn_outliers
        detect_outliers(df)
    end
    if drop_outliers
        df = FlyRL.drop_outliers(df)
    end
    if window > 1
        df = roll(df; filter, window)
    end
    if subsample_resolution > 1
        df = subsample(df, resolution = subsample_resolution)
    end
    if nrow(df) > 0
        for e in encoders
            encode!(e, df)
        end
    end
    df
end
