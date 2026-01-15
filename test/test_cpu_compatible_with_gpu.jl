@testset "CPU_Compatible_GPU" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)

        samples = gensamples(100, 75)
        U       = banded_U(100, 5)
        L       = banded_L(100, 5)

        @testset "uplo = $uplo" for (uplo, pattern) in ((:U, U), (:L, L))

            # Get result from VecchiaMLE on CPU
            model  = VecchiaModel(pattern, samples)
            output = madnlp(model)
            W_cpu  = recover_factor(model, output.solution)

            # Get result from VecchiaMLE on GPU
            model  = VecchiaModel(pattern, CuMatrix(samples))
            output = madnlp(model)
            W_gpu  = recover_factor(model, output.solution)

            @testset norm(W_cpu - SparseMatrixCSC(W_gpu)) ≤ 1e-4
        end
    end
end
