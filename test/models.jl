@testset "softmax" begin
    model = FlyRL.LinearModel(Din = 5, Dout = 3)
    # zero weights
    model.w .= 0
    x = randn(5)
    π = zeros(4)
    FlyRL.softmax!(π, model, x)
    @test sum(π) ≈ 1
    @test π ≈ fill(1/4, 4)
    as = [FlyRL.wsample(π) for _ in 1:10^5]
    @test [length(findall(==(i), as)) for i in 1:4] ./ length(as) ≈ π atol = 1e-1
    # random weights
    model.w .= randn(3, 5)
    wt = [model.w
          zeros(1, 5)]
    FlyRL.softmax!(π, model, x)
    @test π ≈ exp.(wt * x) ./ sum(exp.(wt * x))
    as = [FlyRL.wsample(π) for _ in 1:10^5]
    @test [length(findall(==(i), as)) for i in 1:4] ./ length(as) ≈ π atol = 1e-1
end
