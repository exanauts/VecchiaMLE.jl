@testset "GPU_memory_allocations" begin

    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25] 
    xyGrid = VecchiaMLE.generate_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(n, params, xyGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=GPU)

    model = VecchiaMLE.VecchiaModelGPU(samples, k, xyGrid)
    mems = NLPModelsTest.test_allocs_nlpmodels(model)
    mem_arr = sort(unique([x for x in values(mems) if (!isnan(x) && x > 0.0)]))

    # Check if there isn't one test that is significantly larger than the rest
    if length(mem_arr) < 2
        return 
    end

    @test mems[:obj] == 0.0
    @test mems[:grad!] == 0.0
    @test mems[:cons!] == 0.0
    @test mems[:hess_structure!] == 0.0
    @test mems[:jac_structure!] == 0.0
    @test mems[:jac_coord!] == 0.0
    @test mems[:hess_coord!] == 0.0

    # iterative testing of memory ratios
    for i in eachindex(mem_arr)
        for j in 1:i
            @test mem_arr[i] / mem_arr[j] < 10
            @test mem_arr[i] / mem_arr[j] < 8
            @test mem_arr[i] / mem_arr[j] < 5
            @test mem_arr[i] / mem_arr[j] < 3
        end
    end

end
