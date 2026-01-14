@testset "sparsity_pattern_HNSW" begin
    # Things for model
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)

    MatCov = VecchiaMLE.generate_MatCov(params, ptset)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)

    # Get result from VecchiaMLE NearestNeighbors
    input_NN = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset, sparsitygen=:NN)
    rowsL, colptrL = sparsity_pattern(input_NN)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
    L_NN = recover_factor(colptrL, rowsL, output.solution)

    # Get result from VecchiaMLE HNSW
    input_HNSW = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset, sparsitygen=:HNSW)
    rowsL, colptrL = sparsity_pattern(input_HNSW)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
    L_HNSW = recover_factor(colptrL, rowsL, output.solution)
end
