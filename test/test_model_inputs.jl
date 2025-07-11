@testset "Model_Inputs" begin
    # Things for model
    n = 9
    k = 3
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples; mode=cpu)

    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; ptGrid = ptGrid)
    input.n = 6
    # Test Sanitizaiton input
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.n = 3
    input.k = 20000
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.k = 3
    input.Number_of_Samples = -5 
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.Number_of_Samples = 100
    input.samples = Matrix{Float64}(undef, 1, 1)
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.samples = samples

    ptGrid = [zeros(2) for i in 1:4]
    input.ptGrid = ptGrid
    @test_throws AssertionError VecchiaMLE_Run(input)

    ptGrid = [[0.0] for i in 1:n]
    input.ptGrid = ptGrid
    n = 9
    input.n = 9
    @test_throws AssertionError VecchiaMLE.VecchiaMLE_Run(input)
    input.ptGrid = VecchiaMLE.generate_safe_xyGrid(input.n)
    @test_nowarn VecchiaMLE_Run(input)
end
