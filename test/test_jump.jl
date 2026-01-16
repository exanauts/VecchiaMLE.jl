@testset "Outputs" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)

        samples  = gensamples(100, 75)
        L        = banded_L(100, 3)
        sp       = (L.data.rowval, L.data.colptr)
        nlp      = VecchiaModel(L, samples; lambda=lambda)
        cache    = nlp.cache

        model = Model(MadNLP.Optimizer)
        @variable(model, w[1:(cache.nnzL + cache.n)])
        # Initial L is identity
        for i in cache.colptrL[1:end-1]
            set_start_value(w[i], 1.0)  
        end
        
        # Apply constraints and objective, optimize with JuMP.
        @constraint(model, cons_vecchia(w, cache) .== 0)
        @objective(model, Min, obj_vecchia(w, samples, lambda, cache))
        optimize!(model)
        result_jump = value.(w)

        # Optimize the NLPModels problem.
        result_nlp = madnlp(nlp).solution

        # get and compare the extracted L objects:
        L_jump = recover_factor(nlp, result_jump)
        L_mle  = recover_factor(nlp, result_nlp)

        # matrix result
        @test maximum(abs, L_jump - L_mle) <= 1e-6

        # obj
        @test norm(NLPModels.obj(nlp, result_nlp) - NLPModels.obj(nlp, result_jump)) <= 1e-6 

        # grad
        gx1 = zeros(length(result_nlp))
        gx2 = zeros(length(gx1))
        NLPModels.grad!(nlp, result_nlp, gx1)
        NLPModels.grad!(nlp, result_jump, gx2)

        @test norm(gx1 .- gx2) <= 1e-6

        #cons
        c1 = zeros(nlp.cache.n)
        c2 = zeros(length(c1))
        NLPModels.cons!(nlp, result_nlp, c1)
        NLPModels.cons!(nlp, result_jump, c2)

        @test norm(c1 .- c2) <= 1e-6

        #jprod
        Jv1 = zeros(nlp.cache.n)
        Jv2 = zeros(length(Jv1))
        v1 = ones(nlp.cache.n+nlp.cache.nnzL)
        v2 = ones(nlp.cache.n+nlp.cache.nnzL)
        NLPModels.jprod!(nlp, result_nlp, v1, Jv1)
        NLPModels.jprod!(nlp, result_jump, v2, Jv2)

        @test norm(Jv1 .- Jv2) <= 1e-6 

        #jtprod
        Jtv1 = zeros(nlp.cache.n+nlp.cache.nnzL)
        Jtv2 = zeros(nlp.cache.n+nlp.cache.nnzL)
        v1 = ones(nlp.cache.n)
        v2 = ones(nlp.cache.n)
        NLPModels.jtprod!(nlp, result_nlp, v1, Jtv1)
        NLPModels.jtprod!(nlp, result_jump, v2, Jtv2)
        
        @test norm(Jtv1 .- Jtv2) <= 1e-6 

        #hprod
        y = ones(nlp.cache.n) 
        Hv1 = zeros(nlp.meta.nvar)
        Hv2 = zeros(nlp.meta.nvar)
        v1 = zeros(nlp.meta.nvar)
        v2 = zeros(nlp.meta.nvar)
        NLPModels.hprod!(nlp, result_nlp, y, v1, Hv1)
        NLPModels.hprod!(nlp, result_jump, y, v2, Hv2)
        
        @test norm(Hv1 .- Hv2) <= 1e-6
    end
end
