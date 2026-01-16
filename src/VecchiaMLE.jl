module VecchiaMLE

using KernelAbstractions
using LinearAlgebra
using NLPModels
using Random
using SparseArrays

const KA = KernelAbstractions

# Includes
include("defines.jl")
include("kernels.jl")
include("models/VecchiaModel.jl")
include("models/api.jl")

# Exports
export VecchiaModel, recover_factor

end
