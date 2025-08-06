@testset "Model_Inputs" begin
    # Things for model
    n = 9
    k = 3
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptset)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; mode=cpu)

    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; ptset = ptset)
    input.n = 6
    # Test Sanitizaiton input
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.n = 3
    input.k = 20000
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.k = 3
    input.number_of_samples = -5 
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.number_of_samples = 100
    input.samples = Matrix{Float64}(undef, 1, 1)
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.samples = samples

    ptset = [zeros(2) for i in 1:4]
    input.ptset = ptset
    @test_throws AssertionError VecchiaMLE_Run(input)

    ptset = [[0.0] for i in 1:n]
    input.ptset = ptset
    n = 9
    input.n = 9
    @test_throws AssertionError VecchiaMLE.VecchiaMLE_Run(input)
    input.ptset = VecchiaMLE.generate_safe_xyGrid(input.n)
    @test_nowarn d, L = VecchiaMLE_Run(input)
    d, L = VecchiaMLE_Run(input)

    _, _, vals = findnz(L)
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; x0= ones(length(vals)))
    @test_nowarn VecchiaMLE_Run(input)

    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; x0= zeros(length(vals)))
    @test_warn "User given x0 is not feasible. Setting x0 such that the initial Vecchia approximation is the identity." VecchiaMLE_Run(input)
end
