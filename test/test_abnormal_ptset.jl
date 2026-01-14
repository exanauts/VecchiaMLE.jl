@testset "Abnormal_ptset" begin
    # Things for model
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptset = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptset)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)
    
    for pt in ptset
        pt[1] += randn(Float64) * 1e-2
        pt[2] += randn(Float64) * 1e-2
    end
    
    # Get result from VecchiaMLE cpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    @test_nowarn VecchiaMLE_Run(input)


    ptset = hcat(ptset...)
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    @test_nowarn VecchiaMLE_Run(input)

    ptset = Matrix{Float64}(ptset')
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples; ptset=ptset)
    @test_nowarn VecchiaMLE_Run(input)
end
