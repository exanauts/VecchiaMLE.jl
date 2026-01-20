module NonparametricVecchia

using LinearAlgebra
using NLPModels
using SparseArrays

include("VecchiaModel.jl")
include("api.jl")

export VecchiaModel, recover_factor

end
