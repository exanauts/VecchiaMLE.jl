@testset "CPU_memory_allocations" begin
    # Things for model
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)
    xyGrid = VecchiaMLE.generate_xyGrid(n)

    model_cpu = VecchiaMLE.VecchiaModelCPU(samples, k, xyGrid)
    mems = test_allocs_nlpmodels(model_cpu)

    # Should only expect to see two values, one where we need two vectors and the other where we need one
    mem_arr = sort(unique([mems[i] for i in mems if (!isnan(mems[i]) && mems[i] > 0.0)]))

    @test (length(mem_arr) == 2)
    @test (mem_arr[2] / mem_arr[1] ≈ 2.0)
    
end
