module VecchiaMLE

using BesselK
using CUDA
using CUDA.CUSPARSE
using Distances
using Distributions
using HNSW
using KernelAbstractions
using LinearAlgebra
using NearestNeighbors
using NLPModels
using Random
using SparseArrays
using SpecialFunctions
using Statistics

const KA = KernelAbstractions

# Includes
include("defines.jl")
include("internals.jl")
include("utils.jl")
include("sparsity_pattern.jl")
include("permutations.jl")
include("kernels.jl")
include("models/VecchiaModel.jl")
include("models/api.jl")

# Exports
export VecchiaMLEInput, VecchiaModel
export sparsity_pattern, recover_factor
export generate_samples, generate_MatCov, generate_xyGrid, generate_rectGrid

end
