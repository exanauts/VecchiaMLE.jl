@testset "JuMP_Compatible_GPU" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)
        # Parameters for the model
        n = 9
        k = 3
        number_of_samples = 100
        params = [5.0, 0.2, 2.25, 0.25]
        xyGrid = VecchiaMLE.generate_xyGrid(n)

        MatCov = generate_MatCov(params, xyGrid)
        samples = generate_samples(CuMatrix{Float64}(MatCov), number_of_samples; arch=:gpu)
        Sparsity = sparsity_pattern(xyGrid, k)

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
            model = Model(MadNLP.Optimizer)
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
            samples = CuMatrix{Float64}(samples)
            input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset=xyGrid)
            rowsL, colptrL = sparsity_pattern(input)
            model = VecchiaModel(rowsL, colptrL, samples; lambda, format=:csc, uplo=uplo)
            output = madnlp(model)
            L_mle = recover_factor(model, output.solution)

            @testset norm(SparseMatrixCSC(L_mle) - L_jump) ≤ 1e-6
        end
    end
end
