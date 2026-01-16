@testset "JuMP_Compatible_GPU" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)

        samples = gensamples(100, 75)

        @testset "uplo = $uplo" for uplo in (:L, :U)
            
            pattern = uplo == :L ? banded_L(100, 3) : banded_U(100, 3)
            sp      = (patern.data.rowval, pattern.data.colptr)
            nlp     = VecchiaModel(pattern, samples; lambda=lambda)
            cache   = nlp.cache

            # Model itself
            model = Model(MadNLP.Optimizer)
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

            # compute estimator with JuMP:
            optimize!(model)
            result_jump = value.(w)
            L_jump = recover_factor(nlp, result_jump)

            # compute estimator with VecchiaMLE and GPU:
            model_gpu = VecchiaModel(pattern, CuMatrix{Float64}(samples); lambda=lambda)
            output = madnlp(model_gpu)
            L_mle = recover_factor(model, output.solution)

            @testset norm(SparseMatrixCSC(L_mle) - L_jump) ≤ 1e-6
        end
    end
end
