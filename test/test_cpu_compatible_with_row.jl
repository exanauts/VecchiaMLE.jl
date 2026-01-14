@testset "Row_Compatible_CPU" begin
    # Things for model
    n = 9
    k = 3
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(n)

    MatCov = generate_MatCov(params, xyGrid)
    samples = generate_samples(MatCov, number_of_samples; arch=:cpu)
    Sparsity = sparsity_pattern(xyGrid, k)

    L_row = VecchiaMLE_models(n, k, samples, number_of_samples, Sparsity, "Row")
    L_row = LowerTriangular(L_row)

    # Get result from VecchiaMLE
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=uplo)
    output = madnlp(model)
    L_mle = recover_factor(model, output.solution)
end
