@testset "Different_Metrics" begin
    # Model
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)

    MatCov = generate_MatCov(params, ptset)
    samples = generate_samples(MatCov, number_of_samples; arch=:cpu)
    
    # Get result from VecchiaMLE cpu
    input = VecchiaMLEInput(n, k, samples, number_of_samples; metric=Distances.Euclidean(), ptset=ptset)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
    L_cpu = recover_factor(model, output.solution)

    # Get result from VecchiaMLE cpu
    input = VecchiaMLEInput(n, k, samples, number_of_samples; metric=Distances.Haversine(), ptset=ptset)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
    L_cpu = recover_factor(model, output.solution)
end
