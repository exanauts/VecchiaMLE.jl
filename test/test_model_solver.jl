@testset "different_model_solvers" begin

    samples = gensamples(100, 100)    

    U_pattern = banded_U(100,5)
    model_U   = VecchiaModel(U_pattern, samples)

    @testset "MadNLP" begin
        madnlp(model_U)
    end

    @testset "Ipopt" begin
        ipopt(model_U)
    end

    @testset "Uno" begin
        uno(model_U)
    end
end
