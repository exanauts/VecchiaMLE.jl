
using VecchiaMLE, Vecchia, StaticArrays

# Generate some fake data with an exponential covariance function.
# Note that each _column_ of sim is an iid replicate, in keeping with formatting
# standards of the many R packages for GPs.
pts = rand(SVector{2,Float64}, 1000)
sim = cholesky(Symmetric([exp(-norm(x-y)) for x in pts, y in pts])).L*randn(1000,50)

# Create a VecchiaModel using the options specified by Vecchia.jl, in
# particular choosing an ordering (in this case RandomOrdering()) and a
# conditioning set design (in this case KNNConditioning(10)). This also returns
# the permutation for you to use with subsequent data operations. See the docs
# and myriad extensions of Vecchia.jl for more information on ordering and
# conditioning set design options.
(perm, nlp) = VecchiaModel(pts, sim, RandomOrdering(), KNNConditioning(10))

# Now bring in some optimizers and fit the nonparametric model. This gives a U
# such that Σ^{-1} ≈ U*U', where Σ is the covariance matrix for each column of sim.
using MadNLP
result = madnlp(nlp; tol=1e-10)
recovered_U = recover_factor(nlp, result.solution)

