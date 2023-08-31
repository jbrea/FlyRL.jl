import InteractiveUtils: subtypes
@testset "Encoder API" begin
    track = FlyRL.random_track()
    for enc in subtypes(FlyRL.AbstractEncoder)
        @show enc
        if enc ∈ (FlyRL.VectorEncoder, FlyRL.DurationPerStateEncoder, FlyRL.LevelEncoder)
            e = enc(FlyRL.ShockArmEncoder())
        elseif enc == FlyRL.MarkovKEncoder
            e = enc(2, FlyRL.ArmEncoder())
        elseif enc == FlyRL.DynamicCompressEncoder
            e = enc(:arm, FlyRL.ArmEncoder())
        elseif enc == FlyRL.ColumnPicker
            e = enc(:shock)
        else
            e = enc()
        end
        @test length(FlyRL.labels(e)) == length(FlyRL.levels(e))
        x = FlyRL.encode(e, track)
        if enc == FlyRL.DynamicCompressEncoder
            @test keys(x) == tuple(FlyRL.labels(e)..., :compressed_stream_idxs)
        else
            @test keys(x) == FlyRL.labels(e)
        end
    end
end

@testset "DataFrameRow" begin
    df = DataFrame(x = 100 .+ [0, 2, 4, 8], y = 100 .+ [1, 6, 3, 2], t = [.1, .2, .15, .3])
    encoder = FlyRL.VectorEncoder(FlyRL.AngleEncoder(), FlyRL.FourWallsEncoder())
    x1 = FlyRL.encode(encoder, df)
    FlyRL.encode!(FlyRL.MarkovKEncoder(2, FlyRL.ColumnsPicker(:x, :y, :t), FlyRL.OrientationEncoder()), df)
    df2 = deepcopy(df)
    x2 = FlyRL.encode(encoder, df)
    @test x1 == x2
    x3 = FlyRL.encode.(Ref(encoder), eachrow(df)) |> FlyRL._vecnt2ntvec
    @test x1 == x3
    cols2keep = filter(x -> x ∈ ("x", "y", "t") || !isnothing(match(r"ˌ", "$x")), names(df))
    df.oxˌ1[1] = df.ox[1]
    df.oyˌ1[1] = df.oy[1]
    df3 = select(df, cols2keep)
    x5 = FlyRL.encode.(Ref(encoder), eachrow(df3)) |> FlyRL._vecnt2ntvec
    @test x == x5
end
