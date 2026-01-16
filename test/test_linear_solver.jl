@testset "Linear_Solvers -- $solver" for solver in (:ma27, :ma57)

    samples = gensamples(100,75)
    U       = banded_U(100, 5)
    model   = VecchiaModel(U, samples)

    # hsl
    linear_solver = solver == :ma27 ? MadNLPHSL.Ma27Solver : MadNLPHSL.Ma57Solver
    output = madnlp(model, linear_solver=linear_solver)
    L_hsl = recover_factor(model, output.solution)

    # Umfpack
    output = madnlp(model, linear_solver=MadNLP.UmfpackSolver)
    L_umf = recover_factor(model, output.solution)
end
