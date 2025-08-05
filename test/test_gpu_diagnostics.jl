@testset "GPU_Diagnostics" begin
    # Things for model
    n = 9
    k = 3
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptSet = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptSet)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples; mode=gpu)
    
    # Get result from VecchiaMLE gpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 2; ptSet = ptSet)
    D, L_cpu = VecchiaMLE_Run(input)

    @test (D.MadNLP_iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-5)
    @test (D.LinAlg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)
end
