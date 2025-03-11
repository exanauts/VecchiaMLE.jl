@testset "CUSTOM_KKT_CPU" begin
        # Just to test if it runs. 
        n = 3
        k = 3
        Number_of_Samples = 100
        params = [5.0, 0.2, 2.25, 0.25]
        MatCov = VecchiaMLE.generate_MatCov(n, params)
        samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=GPU)
        
        # Get result from VecchiaMLE GPU
        input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1, 2)
        D, L_cpu = VecchiaMLE_Run(input)
end