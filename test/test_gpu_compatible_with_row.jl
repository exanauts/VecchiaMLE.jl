@testset "Row_Compatible_GPU" begin
    # Things for model
    n = 9
    k = 3
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(Int(sqrt(n)))
    MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples; mode=gpu)
    
    Sparsity = VecchiaMLE.SparsityPattern(xyGrid, k)

    L_row = VecchiaMLE_models(n, k, samples, Number_of_Samples, Sparsity, "Row")
    L_row = LowerTriangular(L_row)


    # Get result from VecchiaMLE
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 2; ptGrid = ptGrid)
    d, L_mle = VecchiaMLE_Run(input)

    errors_row = [VecchiaMLE.KLDivergence(MatCov, L_row), VecchiaMLE.Uni_Error(MatCov, L_row)]
    errors_mle = [VecchiaMLE.KLDivergence(MatCov, L_mle), VecchiaMLE.Uni_Error(MatCov, L_mle)]

    for i in eachindex(errors_mle)
        @test (abs(errors_row[i] ≈ errors_mle[i]) < 1.0)
    end
end
