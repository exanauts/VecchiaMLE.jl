@testset "Abnormal_ptGrid" begin
    # Things for model
    n = 10
    k = 10
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=cpu)
    ptGrid = VecchiaMLE.generate_safe_xyGrid(n)

    for pt in ptGrid
        pt[1] += randn(Float64) * 1e-2
        pt[2] += randn(Float64) * 1e-2
    end
    
    # Get result from VecchiaMLE cpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1)
    _, _ = VecchiaMLE_Run(input; ptGrid)
end