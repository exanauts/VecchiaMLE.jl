@testset "Outputs" begin
    @testset "lambda = $lambda" for lambda in (0.0, 1e-8, 1.0)
        # Parameters for the model
        n = 36
        lambda = 100
        k = 3
        Number_of_Samples = 100
        params = [5.0, 0.2, 2.25, 0.25]
        xyGrid = VecchiaMLE.generate_xyGrid(n)

        MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
        samples = VecchiaMLE.generate_samples(MatCov, Number_of_Samples; mode=:cpu)

        Sparsity = VecchiaMLE.sparsitypattern(xyGrid, k)
        # Model itself
        model = Model(()->MadNLP.Optimizer(max_iter=100, print_level=MadNLP.ERROR))
        cache = create_vecchia_cache_jump(samples, Sparsity, lambda)
        @variable(model, w[1:(cache.nnzL + cache.n)])
        # Initial L is identity
        for i in cache.colptr[1:end-1]
            set_start_value(w[i], 1.0)  
        end
        
        # Apply constraints and objective
        @constraint(model, cons_vecchia(w, cache) .== 0)
        @objective(model, Min, obj_vecchia(w, cache))
        optimize!(model)
        L_jump = sparse(cache.rowsL, cache.colsL, value.(w)[1:cache.nnzL]) 
       
        L_jump = LowerTriangular(L_jump)
        
        # Get result from VecchiaMLE
        input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; lambda=lambda, ptset=xyGrid)
        d, L_mle = VecchiaMLE_Run(input)
        L_mle = LowerTriangular(L_mle)
        # get model from VecchiaMLE
        nlp = VecchiaMLE.get_vecchia_model(input)
        _, _, mle_vals = findnz(sparse(L_mle))
        _, _, jump_vals = findnz(sparse(L_jump))
        append!(mle_vals, [log(x) for x in mle_vals[nlp.cache.diagL]])
        append!(jump_vals, [log(x) for x in jump_vals[nlp.cache.diagL]])

        # obj
        @test norm(VecchiaMLE.NLPModels.obj(nlp, mle_vals) - VecchiaMLE.NLPModels.obj(nlp, jump_vals)) <= 1e-6 

        # grad
        gx1 = zeros(length(mle_vals))
        gx2 = zeros(length(gx1))
        VecchiaMLE.NLPModels.grad!(nlp, mle_vals, gx1)
        VecchiaMLE.NLPModels.grad!(nlp, jump_vals, gx2)

        @test norm(gx1 .- gx2) <= 1e-6

        #cons
        c1 = zeros(nlp.cache.n)
        c2 = zeros(length(c1))
        VecchiaMLE.NLPModels.cons!(nlp, mle_vals, c1)
        VecchiaMLE.NLPModels.cons!(nlp, jump_vals, c2)

        @test norm(c1 .- c2) <= 1e-6

        #jprod
        Jv1 = zeros(nlp.cache.n)
        Jv2 = zeros(length(Jv1))
        v1 = ones(nlp.cache.n+nlp.cache.nnzL)
        v2 = ones(nlp.cache.n+nlp.cache.nnzL)
        VecchiaMLE.NLPModels.jprod!(nlp, mle_vals, v1, Jv1)
        VecchiaMLE.NLPModels.jprod!(nlp, jump_vals, v2, Jv2)

        @test norm(Jv1 .- Jv2) <= 1e-6 

        #jtprod
        Jtv1 = zeros(nlp.cache.n+nlp.cache.nnzL)
        Jtv2 = zeros(nlp.cache.n+nlp.cache.nnzL)
        v1 = ones(nlp.cache.n)
        v2 = ones(nlp.cache.n)
        VecchiaMLE.NLPModels.jtprod!(nlp, mle_vals, v1, Jtv1)
        VecchiaMLE.NLPModels.jtprod!(nlp, jump_vals, v2, Jtv2)
        
        @test norm(Jtv1 .- Jtv2) <= 1e-6 

        #hprod
        y = ones(nlp.cache.n) 
        Hv1 = zeros(nlp.meta.nvar)
        Hv2 = zeros(nlp.meta.nvar)
        v1 = zeros(nlp.meta.nvar)
        v2 = zeros(nlp.meta.nvar)
        VecchiaMLE.NLPModels.hprod!(nlp, mle_vals, y, v1, Hv1)
        VecchiaMLE.NLPModels.hprod!(nlp, jump_vals, y, v2, Hv2)
        
        @test norm(Hv1 .- Hv2) <= 1e-6
    end
end
