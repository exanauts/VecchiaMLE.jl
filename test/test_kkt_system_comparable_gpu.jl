@testset "COMPARE_KKT_GPU" begin
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=GPU)
    
    # Get the result from default KKT
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 2, 1)
    D_def, _ = VecchiaMLE_Run(input)

    # Get the result from custom KKT
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 2, 2)
    D_cus, _ = VecchiaMLE_Run(input)

    # Don't know how tight we want the values to be
    @test (abs(D_def.normed_constraint_value - D_cus.normed_constraint_value) < 1e-6)
    @test (abs(D_def.normed_grad_value - D_cus.normed_grad_value) < 1e-6)
end