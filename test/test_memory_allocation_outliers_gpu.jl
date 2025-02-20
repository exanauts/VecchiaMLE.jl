@testset "GPU_memory_allocations" begin
    # Things for model
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)
    xyGrid = VecchiaMLE.generate_xyGrid(n)

    # Get result from VecchiaMLE CPU
    
    model_cpu = VecchiaMLE.VecchiaModelGPU(samples, k, xyGrid)

    # Should only expect to see two values, one where we need two vectors and the other where we need one
    mem_arr = sort(unique([x for x in values(model_cpu) if (!isnan(x) && x > 0.0)]))

    @test (length(mem_arr) == 2)
    @test (mem_arr[2] / mem_arr[1] ≈ 2.0)
    
end
