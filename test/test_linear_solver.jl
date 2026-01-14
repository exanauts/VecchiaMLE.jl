@testset "Linear_Solvers -- $solver" for solver in (:ma27, :ma57)
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = generate_safe_xyGrid(n)
    MatCov = generate_MatCov(params, ptset)
    samples = generate_samples(MatCov, number_of_samples)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)

    # hsl
    linear_solver = solver == :ma27 ? MadNLPHSL.Ma27Solver : MadNLPHSL.Ma57Solver
    output = madnlp(model, linear_solver=linear_solver)
    L_hsl = recover_factor(model, output.solution)

    # Umfpack
    output = madnlp(model, linear_solver=MadNLP.UmfpackSolver)
    L_umf = recover_factor(model, output.solution)
end
