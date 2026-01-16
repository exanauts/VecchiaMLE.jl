@testset "COO vs CSC constructor" begin

    samples = gensamples(100, 75)    

    U_pattern   = banded_U(100,5)
    model_U_csc = VecchiaModel(U_pattern, samples)

    (I, J, _)   = findnz(U_pattern.data)
    model_U_coo = VecchiaModel(I, J, samples; uplo=:U)

    res_csc = madnlp(model_U_csc; tol=1e-10).solution
    res_coo = madnlp(model_U_coo; tol=1e-10).solution
    @test res_csc ≈ res_coo

end
