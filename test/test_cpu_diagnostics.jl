@testset "CPU_Diagnostics" begin
    # Things for model
    n = 3
    k = 3
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=COMPUTE_MODE_CPU)
    
    # Get result from VecchiaMLE CPU
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1)
    D, L_cpu = VecchiaMLE_Run(input)

    @test (D.MadNLP_iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-7)
    @test (D.LinAlg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)
end
