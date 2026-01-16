@testset "Different_Metrics" begin

    samples = gensamples(100,75)
    U = banded_U(100,5)
    model = VecchiaModel(U, samples)
    output = madnlp(model)
    L_cpu = recover_factor(model, output.solution)

    # Get result from VecchiaMLE cpu
    input = VecchiaMLEInput(n, k, samples, number_of_samples; metric=Distances.Haversine(), ptset=ptset)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
    L_cpu = recover_factor(model, output.solution)
end
