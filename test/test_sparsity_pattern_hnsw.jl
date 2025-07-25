@testset "SparsityPattern_HNSW" begin
    # Things for model
    n = 100
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples; mode=cpu)
    
    # Get result from VecchiaMLE NearestNeighbors
    inputNN = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; ptGrid = ptGrid, sparsityGeneration=VecchiaMLE.NN)
    D, L_NN = VecchiaMLE_Run(inputNN)

    @test (D.MadNLP_iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-5)
    @test (D.LinAlg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)

    # Get result from VecchiaMLE HNSW
    inputHNSW = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; ptGrid = ptGrid, sparsityGeneration=VecchiaMLE.HNSW)
    D, L_HNSW = VecchiaMLE_Run(inputNN)

    @test (D.MadNLP_iterations ≥ 0)
    @test (D.normed_constraint_value < 1e-3)
    @test (D.normed_grad_value < 1e-5)
    @test (D.LinAlg_solve_time > 0.0)
    @test (D.solve_model_time > 0.0)
    @test (D.create_model_time > 0.0)


    errors_nn = [VecchiaMLE.KLDivergence(MatCov, L_NN), VecchiaMLE.Uni_Error(MatCov, L_NN)]
    errors_hsnw = [VecchiaMLE.KLDivergence(MatCov, L_HNSW), VecchiaMLE.Uni_Error(MatCov, L_HNSW)]

    for i in eachindex(errors_hsnw)
        @test abs(errors_nn[i] - errors_hsnw[i]) < 1e-6 # Don't know a good bound
    end
end
