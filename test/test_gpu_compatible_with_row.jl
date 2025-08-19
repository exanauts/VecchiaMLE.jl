@testset "Row_Compatible_GPU" begin
    # Things for model
    n = 9
    k = 3
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(n)

    MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; mode=:gpu)
    
    Sparsity = VecchiaMLE.sparsitypattern(xyGrid, k)

    L_row = VecchiaMLE_models(n, k, samples, number_of_samples, Sparsity, "Row")
    L_row = LowerTriangular(L_row)


    # Get result from VecchiaMLE
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 2; ptset = ptset)
    d, L_mle = VecchiaMLE_Run(input)

    errors_row = [VecchiaMLE.KLDivergence(MatCov, L_row), VecchiaMLE.uni_error(MatCov, L_row)]
    errors_mle = [VecchiaMLE.KLDivergence(MatCov, L_mle), VecchiaMLE.uni_error(MatCov, L_mle)]

    for i in eachindex(errors_mle)
        @test (abs(errors_row[i] ≈ errors_mle[i]) < 1.0)
    end
end
