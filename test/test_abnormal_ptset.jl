@testset "Abnormal_ptset" begin
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = generate_MatCov(params, ptset)
    samples = generate_samples(MatCov, number_of_samples; arch=:cpu)
    
    for pt in ptset
        pt[1] += randn(Float64) * 1e-2
        pt[2] += randn(Float64) * 1e-2
    end
    
    # Get result from VecchiaMLE cpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)

    ptset = hcat(ptset...)
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)

    ptset = Matrix{Float64}(ptset')
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)
    output = madnlp(model)
end
