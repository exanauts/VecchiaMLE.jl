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
            input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
            rowsL, colptrL = sparsity_pattern(input)
            model = VecchiaModel(rowsL, colptrL, samples; lambda, format=:csc, uplo=uplo)
            output = madnlp(model)
            L_cpu = recover_factor(model, output.solution)

            # Get result from VecchiaMLE on GPU
            input = VecchiaMLE.VecchiaMLEInput(n, k, CuMatrix(samples), number_of_samples; arch=:gpu, ptset=ptset)
            model = VecchiaModel(rowsL, colptrL, samples; lambda, format=:csc, uplo=uplo)
            output = madnlp(model)
            L_gpu = recover_factor(model, output.solution)

            @testset norm(L_cpu - SparseMatrixCSC(L_gpu)) ≤ 1e-4
        end
    end
end
