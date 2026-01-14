@testset "Row_Compatible_GPU" begin
    # Things for model
    n = 9
    k = 3
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = generate_xyGrid(n)

    MatCov = generate_MatCov(params, xyGrid)
    samples = generate_samples(MatCov, number_of_samples; arch=:gpu)
    Sparsity = sparsity_pattern(xyGrid, k)

    L_row = VecchiaMLE_models(n, k, samples, number_of_samples, Sparsity, "Row")
    L_row = LowerTriangular(L_row)

    # Get result from VecchiaMLE
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; arch=:gpu, ptset=ptset)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
    L_mle = recover_factor(model, output.solution)

    @testset norm(SparseMatrixCSC(L_mle) - L_row) ≤ 1e-6
end
