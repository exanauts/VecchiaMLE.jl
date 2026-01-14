```julia
using VecchiaMLE
using LinearAlgebra
using SparseArrays

n = 400
number_of_samples = 100

params = [5.0, 0.2, 2.25, 0.25]
ptset = generate_safe_xyGrid(n)
MatCov = generate_MatCov(params, ptset)
samples = generate_samples(MatCov, number_of_samples)

P = ones(n, n)
P = tril(P)
P = sparse(P)
I, J, V = findnz(P)
nlp_L = VecchiaModel(I, J, samples; format=:coo, uplo=:L)

P = ones(n, n)
P = triu(P)
P = sparse(P)
I, J, V = findnz(P)
nlp_U = VecchiaModel(I, J, samples; format=:coo, uplo=:U)
```
