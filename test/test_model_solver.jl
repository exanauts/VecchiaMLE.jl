@testset "different_model_solvers" begin

    samples = gensamples(100, 75)    

    U_pattern   = banded_U(100,5)
    model_U_csc = VecchiaModel(U_pattern, samples)

    @testset "MadNLP" begin
        madnlp(model_U_csc)
    end

    @testset "Ipopt" begin
        ipopt(model_U_csc)
    end

    @testset "Uno" begin
        uno(model_U_csc)
    end

end
