@testset "GPU_Diagnostics" begin
    # Things for model
    n = 9
    k = 3
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)

    MatCov = VecchiaMLE.generate_MatCov(params, ptset)
    samples = VecchiaMLE.generate_samples(CuMatrix{Float64}(MatCov), number_of_samples; arch=:gpu)
    
    # Get result from VecchiaMLE gpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; arch=:gpu, ptset = ptset)
    D, L_cpu = VecchiaMLE_Run(input)

    @test (D.iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-5)
    @test (D.linalg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)
end
