@testset "CPU_memory_allocations" begin
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = generate_xyGrid(n)
    MatCov = generate_MatCov(params, xyGrid)
    samples = generate_samples(MatCov, number_of_samples; arch=:cpu)

    iVecchiaMLE = VecchiaMLEInput(n, k, samples, number_of_samples; ptset=xyGrid)
    rowsL, colptrL = sparsity_pattern(iVecchiaMLE)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    mems = NLPModelsTest.test_allocs_nlpmodels(model)

    @test mems[:obj] == 0.0
    @test mems[:grad!] == 0.0
    @test mems[:cons!] == 0.0
    @test mems[:hess_structure!] == 0.0
    @test mems[:jac_structure!] == 0.0
    @test mems[:jac_coord!] == 0.0
    @test mems[:hess_coord!] == 0.0
    @test mems[:hess_lag_coord!] == 0.0
end
