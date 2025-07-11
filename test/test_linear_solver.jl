@testset "Linear_Solvers -- $solver" for solver in (:ma27, :ma57)

    # Things for model
    n = 100
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples)

    input = VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; ptGrid = ptGrid)
    linear_solver = solver == :ma27 ? MadNLPHSL.Ma27Solver : MadNLPHSL.Ma57Solver

    model = VecchiaMLE.get_vecchia_model(input)
    output = madnlp(model,
        linear_solver=linear_solver,
        print_level=input.pLevel
    )

    valsL = Vector{Float64}(output.solution[1:model.cache.nnzL])
    rowsL = Vector{Int}(model.cache.rowsL)
    colsL = Vector{Int}(model.cache.colsL)
    L_hsl = LowerTriangular(sparse(rowsL, colsL, valsL))

    # Umfpack
    model = VecchiaMLE.get_vecchia_model(input)
    output = madnlp(model,
        linear_solver=MadNLP.UmfpackSolver,
        print_level=input.pLevel
    )

    valsL = Vector{Float64}(output.solution[1:model.cache.nnzL])
    rowsL = Vector{Int}(model.cache.rowsL)
    colsL = Vector{Int}(model.cache.colsL)

    L_umf = LowerTriangular(sparse(rowsL, colsL, valsL))

    # Then check if we have the same result.
    kl_hsl = VecchiaMLE.KLDivergence(MatCov, L_hsl)
    kl_umf = VecchiaMLE.KLDivergence(MatCov, L_umf)

    @test isnan(kl_hsl) == false
    @test isnan(kl_umf) == false
    @test kl_hsl < 100
    @test kl_umf < 100

    @test (abs(kl_hsl - kl_umf) < 1e-6) # Don't know a good bound for this.
end
