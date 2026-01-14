@testset "different_model_solvers" begin
    # Generating samples
    n = 100
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
    samples = VecchiaMLE.generate_samples(MatCov, Number_of_Samples; arch=:cpu)

    # Only check if the inputs do not cause issues. TODO: Resolve kwargs for the different optimizers (print_level, ...)
    input_madnlp = VecchiaMLEInput(n, k, samples, Number_of_Samples; ptset=xyGrid, solver=:madnlp)
    input_knitro = VecchiaMLEInput(n, k, samples, Number_of_Samples; ptset=xyGrid, solver=:knitro)
    input_ipopt  = VecchiaMLEInput(n, k, samples, Number_of_Samples; ptset=xyGrid, solver=:ipopt)    
end