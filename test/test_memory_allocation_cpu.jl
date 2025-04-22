@testset "CPU_memory_allocations" begin
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(n, params, xyGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=cpu)
    iVecchiaMLE = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 1, 5; ptGrid=xyGrid)
    model = VecchiaMLE.VecchiaModelCPU(samples, iVecchiaMLE)
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
