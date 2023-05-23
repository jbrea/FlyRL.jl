@testset "Encoder API" begin
    track = FlyRL.random_track()
    for enc in subtypes(FlyRL.AbstractEncoder)
        @show enc
        if enc == FlyRL.VectorEncoder
            e = enc(FlyRL.ArmEncoder())
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
