@testset "CPU_Compatible_GPU" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 100.0)
        # Parameters for the model
        n = 9
        k = 3
        number_of_samples = 100
        params = [5.0, 0.2, 2.25, 0.25]
        ptset = VecchiaMLE.generate_safe_xyGrid(n)
        MatCov = VecchiaMLE.generate_MatCov(params, ptset)
        samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; mode=cpu)

        # Get result from VecchiaMLE on CPU
        input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; lambda=lambda, ptset=ptset)
        _, L_cpu = VecchiaMLE_Run(input)

        # Get result from VecchiaMLE on GPU
        input = VecchiaMLE.VecchiaMLEInput(n, k, CuMatrix(samples), number_of_samples, 5, 2; lambda=lambda, ptset=ptset)
        _, L_gpu = VecchiaMLE_Run(input)

        @testset norm(L_cpu - SparseMatrixCSC(L_gpu)) ≤ 1e-4

        if lambda == 0.0
            errors_cpu = [VecchiaMLE.KLDivergence(MatCov, L_cpu), VecchiaMLE.uni_error(MatCov, L_cpu)]
            errors_gpu = [VecchiaMLE.KLDivergence(MatCov, L_gpu), VecchiaMLE.uni_error(MatCov, L_gpu)]

            for i in eachindex(errors_gpu)
                @test (abs(errors_cpu[i] - errors_gpu[i]) < 0.01)
            end
        end
    end
end
