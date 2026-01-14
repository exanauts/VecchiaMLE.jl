@testset "CPU_Compatible_GPU" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)
        # Parameters for the model
        n = 9
        k = 3
        number_of_samples = 100
        params = [5.0, 0.2, 2.25, 0.25]
        ptset = VecchiaMLE.generate_safe_xyGrid(n)
        MatCov = VecchiaMLE.generate_MatCov(params, ptset)
        samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)

        @testset "uplo = $uplo" for uplo in (:L, :U)
            # Get result from VecchiaMLE on CPU
            input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; lambda=lambda, ptset=ptset, uplo=uplo)
            _, L_cpu = VecchiaMLE_Run(input)

            # Get result from VecchiaMLE on GPU
            input = VecchiaMLE.VecchiaMLEInput(n, k, CuMatrix(samples), number_of_samples; arch=:gpu, lambda=lambda, ptset=ptset, uplo=uplo)
            _, L_gpu = VecchiaMLE_Run(input)

            @testset norm(L_cpu - SparseMatrixCSC(L_gpu)) ≤ 1e-4
        end
    end
end
