@testset "Linear_Solvers" begin

    # Things for model
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptset)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset = ptset, linear_solver=:ma27)
    d, L_ma27 = VecchiaMLE_Run(input)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset = ptset, linear_solver=:ma57)
    d, L_ma57 = VecchiaMLE_Run(input)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset = ptset, linear_solver=:ma86)
    d, L_ma86 = VecchiaMLE_Run(input)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset = ptset, linear_solver=:ma97)
    d, L_ma97 = VecchiaMLE_Run(input)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset = ptset, linear_solver=:umfpack)
    d, L_umf = VecchiaMLE_Run(input)



    # Then check if we have the same result.
    kl_umf = VecchiaMLE.KLDivergence(MatCov, L_umf)
    kl_27 = VecchiaMLE.KLDivergence(MatCov, L_27)
    kl_57 = VecchiaMLE.KLDivergence(MatCov, L_57)
    kl_86 = VecchiaMLE.KLDivergence(MatCov, L_86)
    kl_97 = VecchiaMLE.KLDivergence(MatCov, L_97)
    
    @test isnan(kl_hsl) == false
    @test isnan(kl_umf) == false
    
    @test kl_umf < 100
    @test kl_27 < 100
    @test kl_57 < 100
    @test kl_86 < 100
    @test kl_97 < 100

    @test (abs(kl_27 - kl_umf) < 1e-6) # Don't know a good bound for this.
end
