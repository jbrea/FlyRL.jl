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
num_outliers(x, y) = sum(==(false), in_maze.(x, y))
function fine_tune_calibration!(df, x, y)
    dxmin = 0
    dymin = 0
    outliersmin = num_outliers(x, y)
    for dx in -4:4
        for dy in -4:4
            outliers = num_outliers(x .+ dx, y .+ dy)
            if outliers < outliersmin
                outliersmin = outliers
                dxmin = dx
                dymin = dy
            elseif outliers == outliersmin && abs(dx) + abs(dy) < abs(dxmin) + abs(dymin)
                dxmin = dx
                dymin = dy
            end
        end
    end
    df.x .+= dxmin
    df.y .+= dymin
    df
end
function calibrate!(df, i = 0;
                    coords = union(tuple.(df.x, df.y)),
                    x = first.(coords), y = last.(coords),
                    max_iterations = 10)
    offsets = offset_to_closest.(x, y)
    dx = round(Int, median(first.(offsets)))
    dy = round(Int, median(last.(offsets)))
    x .+= dx
    y .+= dy
    Δ = abs(dx) + abs(dy)
    if Δ == 0 || i == max_iterations
        df.x .+= x[1] - df.x[1]
        df.y .+= y[1] - df.y[1]
        return fine_tune_calibration!(df, x, y)
    end
    calibrate!(df, i+1; coords, x, y, max_iterations)
end

function drop_initial_outliers(df; starttime = 0, tolerance = 1)
    starti = findfirst(≥(starttime), df.t)
    tol = tolerance
    for (i, r) ∈ Iterators.drop(pairs(eachrow(df)), starti)
        if !in_maze(r.x, r.y)
            tol = tolerance
        elseif tol == 0
            starti = i - tolerance
            break
        else
            tol -= 1
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
    idxs = findall((!).(in_maze.(df.x, df.y)))
    outliers = union([(df.x[i], df.y[i]) for i in idxs])
#     push!(Main.outliers, (metadata(df)["filename"], outliers))
    if length(idxs) > 0
        @warn "Detected outliers in rows $idxs at positions $outliers"
    end
end

"""
$SIGNATURES

Preprocess data frame `df`.
"""
function preprocess(
    df;
    filter = rollmedian,
    window = 1,
    warn_outliers = false,
    drop_outliers = true,
    encoders = (ArmEncoder(with_outliers = drop_outliers == false),
                ShockArmEncoder(with_outliers = drop_outliers == false)),
    subsample_resolution = 1,
    calibrate = true,
    relative_time = true,
    maxtime = 1200
)
    df = df[(df.x .> 0) .& (df.y .> 0), :]
    nrow(df) == 0 && return df
    if calibrate
        calibrate!(df)
    end
    df = drop_initial_outliers(df)
    if relative_time
        df.t .-= df.t[1]
    end
    if maxtime < Inf
        df = df[df.t .< maxtime, :]
    end
    if warn_outliers
        detect_outliers(df)
    end
    if drop_outliers
        df = YMaze.drop_outliers(df)
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

"""
$SIGNATURES

Read track `f` in directory `root` and corresponding time, shock and pattern file.
Returns a `DataFrame`. `kwargs` are passed to [`preprocess`](@ref).
"""
function read(root, f;
              preprocess = true, load_arm = true,
              load_shock = true, load_pattern = true,
              timefile = joinpath(root, replace(f, "track" => "time")),
              shockfile = joinpath(root, replace(f, "track" => "shock")),
              patternfile = joinpath(root, replace(f, "track" => "pattern")),
              armfile = joinpath(root, replace(f, "track" => "arm")),
              kwargs...)
    track = CSV.read(joinpath(root, f), DataFrame, header = [:x, :y])
    t = CSV.read(timefile, DataFrame, header = [:time])
    track.t = (t.time .- t.time[1]) ./ 1e9 # time in s
    if load_shock
        s = CSV.read(shockfile, DataFrame, header = [:shock],)
        track.raw_shock = s.shock .== "X"
    end
    if load_pattern
        p = CSV.read(patternfile, DataFrame, header = [:pattern],)
        track.pattern = p.pattern
    end
    if load_arm
        p = CSV.read(armfile, DataFrame, header = [:arm],)
        track.raw_arm = p.arm
    end
    metadata!(track, "filename", joinpath(root, f), style = :note)
    if preprocess
        track = YMaze.preprocess(track; kwargs...)
    end
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

