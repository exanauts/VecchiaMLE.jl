@testset "CPU_Compatible_GPU" begin
    # Things for model
    n = 9
    k = 3
    Number_of_Samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, ptGrid)
    samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples; mode=cpu)
    
    # Get result from VecchiaMLE cpu
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; ptGrid=ptGrid)
    _, L_cpu = VecchiaMLE_Run(input)

    input = VecchiaMLE.VecchiaMLEInput(n, k, CuMatrix(samples), Number_of_Samples, 5, 2; ptGrid=ptGrid)
    _, L_gpu = VecchiaMLE_Run(input)

    errors_cpu = [VecchiaMLE.KLDivergence(MatCov, L_cpu), VecchiaMLE.Uni_Error(MatCov, L_cpu)]
    errors_gpu = [VecchiaMLE.KLDivergence(MatCov, L_gpu), VecchiaMLE.Uni_Error(MatCov, L_gpu)]

    for i in eachindex(errors_gpu)
        @test (abs(errors_cpu[i] - errors_gpu[i]) < 0.01)
    end
end
