@testset "JuMP_Compatible_CPU" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)
        # Parameters for the model
        n = 9
        k = 3
        number_of_samples = 100
        params = [5.0, 0.2, 2.25, 0.25]
        xyGrid = VecchiaMLE.generate_xyGrid(n)

        MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
        samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)
        Sparsity = VecchiaMLE.sparsitypattern(xyGrid, k)

        @testset "uplo = $uplo" for uplo in (:L, :U)
            if uplo == :U
                rows, colptr = Sparsity
                nnzL = length(rows)
                nzval = ones(Float64, nnzL)
                L = SparseMatrixCSC(n, n, colptr, rows, nzval)
                U = sparse(L')
                Sparsity = (U.rowval, U.colptr)
            end

            # Model itself
            model = Model(()->MadNLP.Optimizer(max_iter=200, print_level=MadNLP.ERROR))
            cache = create_vecchia_cache_jump(samples, Sparsity, lambda, uplo)
            @variable(model, w[1:(cache.nnzL + cache.n)])
            # Initial L is identity
            if uplo == :L
                for i in cache.colptrL[1:end-1]
                    set_start_value(w[i], 1.0)
                end
            else
                for i in cache.colptrL[2:end]
                    set_start_value(w[i-1], 1.0)
                end
            end
            # Apply constraints and objective
            @constraint(model, cons_vecchia(w, cache) .== 0)
            @objective(model, Min, obj_vecchia(w, cache))

            optimize!(model)
            L_jump = SparseMatrixCSC(cache.n, cache.n, cache.colptrL, cache.rowsL, value.(w)[1:cache.nnzL])
            L_jump = uplo == :L ? LowerTriangular(L_jump) : UpperTriangular(L_jump)

            # Get result from VecchiaMLE
            input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; lambda=lambda, ptset=xyGrid, uplo=uplo)
            d, L_mle = VecchiaMLE_Run(input)
            L_mle = uplo == :L ? LowerTriangular(L_mle) : UpperTriangular(L_mle)

            @testset norm(L_mle - L_jump) ≤ 1e-6

            if uplo == :L
                errors_jump = [VecchiaMLE.KLDivergence(MatCov, L_jump), VecchiaMLE.uni_error(MatCov, L_jump)]
                errors_mle = [VecchiaMLE.KLDivergence(MatCov, L_mle), VecchiaMLE.uni_error(MatCov, L_mle)]

                for i in eachindex(errors_mle)
                    @test (errors_jump[i] ≈ errors_mle[i])
                end
            end
        end
    end
end
