@testset "Model_Inputs" begin
    # Parameters for the model
    n = 9
    k = 3
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)

    MatCov = VecchiaMLE.generate_MatCov(params, ptset)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)

    # default input
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset = ptset)
    
    # test dimension (n value)
    input.n = -1
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.n = 9
    
    # test conditioning neighbors
    input.k = 200
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.k = 3
    
    # test number_of_samples
    input.number_of_samples = -5 
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.number_of_samples = 100
    
    # test samples matrix
    input.samples = Matrix{Float64}(undef, 1, 1)
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.samples = samples

    # test length(ptset) == n
    input.ptset = [zeros(2) for i in 1:4]
    @test_throws AssertionError VecchiaMLE_Run(input)
    input.ptset = ptset

    # test it actually runs
    d, L = VecchiaMLE_Run(input)

    # test inital condition
    L_csc = SparseMatrixCSC(L)
    rowsL = L_csc.rowval
    colptr = L_csc.colptr
    vals = L_csc.nzval
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; x0= zeros(length(vals)))
    @test_warn "User given x0 is not feasible. Setting x0 such that the initial Vecchia approximation is the identity." VecchiaMLE_Run(input)
    
    colsL = similar(rowsL)
    for j in 1:length(colptr)-1
        idx_range = colptr[j]:(colptr[j+1]-1)
        colsL[idx_range] .= j
    end

    # rows, columns
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; rowsL = rowsL, colsL = colsL, colptrL = colptr)
    @test_nowarn VecchiaMLE_Run(input)

    # columns swapped with rows (will run technically, just worse than other)
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, 5, 1; rowsL = colsL, colsL = rowsL, colptrL = colptr)
    VecchiaMLE_Run(input)

    # test solver_tol
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; solver_tol = -1.0) 
    @test_throws AssertionError VecchiaMLE_Run(input)

    # test if minimal inputs passes
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples)
    VecchiaMLE_Run(input)

    # Check print level
    @test_throws ErrorException VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, :BLANK, :cpu)

    # Check architecture
    @test_throws ErrorException VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, :VERROR, :BLANK)

    # check solvers
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, :VERROR, :cpu; solver=:BLANK)
    @test_throws AssertionError VecchiaMLE_Run(input) 

    # check sparsity gen
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples, :VERROR, :cpu; sparsitygen=:BLANK)
    @test_throws AssertionError VecchiaMLE_Run(input) 

end
