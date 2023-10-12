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
        calibrate!(df)
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
