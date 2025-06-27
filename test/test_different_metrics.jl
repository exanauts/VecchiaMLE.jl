@testset "Different_Metrics" begin
    # Things for model
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=cpu)
    
    # Get result from VecchiaMLE cpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; metric = Distances.Euclidean())
    D, L_cpu = VecchiaMLE_Run(input)

    @test (D.MadNLP_iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-5)
    @test (D.LinAlg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)

        # Get result from VecchiaMLE cpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; metric = Distances.Haversine())
    D, L_cpu = VecchiaMLE_Run(input)

    @test (D.MadNLP_iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-5)
    @test (D.LinAlg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)
end
