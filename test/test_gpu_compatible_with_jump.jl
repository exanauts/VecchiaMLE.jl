@testset "JuMP_Compatible_GPU" begin
    # Things for model
    n = 9
    k = 3
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples; mode=cpu)
    
    Sparsity = VecchiaMLE.SparsityPattern(xyGrid, k)

    # Model itself
    model = Model(()->MadNLP.Optimizer(max_iter=100, print_level=MadNLP.ERROR))
    cache = create_vecchia_cache_jump(samples, Sparsity)
    @variable(model, w[1:(cache.nnzL + cache.n)])
    # Initial L is identity
    for i in cache.colptr[1:end-1]
        set_start_value(w[i], 1.0)  
    end
    # Apply constraints and objective
    @constraint(model, cons_vecchia(w, cache) .==0)
    @objective(model, Min, obj_vecchia(w, cache))

    optimize!(model)
    L_jump = sparse(cache.rowsL, cache.colsL, value.(w)[1:cache.nnzL]) 
   
    L_jump = LowerTriangular(L_jump)

    # Get result from VecchiaMLE
    samples = CuMatrix{Float64}(samples)
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 2; ptSet = xyGrid)
    d, L_mle = VecchiaMLE_Run(input)

    errors_jump = [VecchiaMLE.KLDivergence(MatCov, L_jump), VecchiaMLE.Uni_Error(MatCov, L_jump)]
    errors_mle = [VecchiaMLE.KLDivergence(MatCov, L_mle), VecchiaMLE.Uni_Error(MatCov, L_mle)]

    for i in eachindex(errors_mle)
        @test (abs(errors_jump[i] - errors_mle[i]) < 0.01)
    end
end
