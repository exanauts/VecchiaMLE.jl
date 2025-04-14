@testset "Linear_Solvers" begin

    @test LIBHSL_isfunctional()
    # Things for model
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)
    
    input = VecchiaMLEInput(n, k, samples, Number_of_Samples)

    model = VecchiaMLE.get_vecchia_model(input)
    output = madnlp(model, 
        linear_solver=MadNLPHSL.Ma57Solver,
        print_level=VecchiaMLE.MadNLP_Print_Level(input.MadNLP_print_level)
    )


    valsL = Vector{Float64}(output.solution[1:model.cache.nnzL])
    rowsL = Vector{Int}(model.cache.rowsL)
    colsL = Vector{Int}(model.cache.colsL)

    L_ma57 = LowerTriangular(sparse(rowsL, colsL, valsL))

    # Umfpack
    model = VecchiaMLE.get_vecchia_model(input)
    output = madnlp(model, 
        linear_solver=MadNLP.UmfpackSolver,
        print_level=VecchiaMLE.MadNLP_Print_Level(input.MadNLP_print_level)
    )


    valsL = Vector{Float64}(output.solution[1:model.cache.nnzL])
    rowsL = Vector{Int}(model.cache.rowsL)
    colsL = Vector{Int}(model.cache.colsL)

    L_umf = LowerTriangular(sparse(rowsL, colsL, valsL))


    # Then check if we have the same result. 
    kl_57 = VecchiaMLE.KLDivergence(MatCov, L_ma57)
    kl_umf = VecchiaMLE.KLDivergence(MatCov, L_umf)
    
    @test isnan(kl_57) == false
    @test isnan(kl_umf) == false
    @test kl_57 < 100
    @test kl_umf < 100

    @test (abs(kl_57 - kl_umf) < 1e-6) # Don't know a good bound for this. 


end