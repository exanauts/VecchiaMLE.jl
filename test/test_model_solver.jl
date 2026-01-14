@testset "different_model_solvers" begin
    # Generating samples
    n = 100
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]
    xyGrid = VecchiaMLE.generate_xyGrid(n)
    MatCov = VecchiaMLE.generate_MatCov(params, xyGrid)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)

    input = VecchiaMLEInput(n, k, samples, number_of_samples; ptset=xyGrid)
    rowsL, colptrL = sparsity_pattern(input)
    model = VecchiaModel(rowsL, colptrL, samples; format=:csc, uplo=:L)

    @testset "MadNLP" begin
        madnlp(model)
    end

    @testset "Ipopt" begin
        ipopt(model)
    end

    @testset "Uno" begin
        uno(model)
    end
end
