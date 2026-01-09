module VecchiaMLE

# External Packages
using Random
using NLPModels, LinearAlgebra
using Distances
using NearestNeighbors
using MadNLP, MadNLPGPU, MadNLPHSL
using SparseArrays 
using CUDA
using BesselK, SpecialFunctions
using Distributions, Statistics
using KernelAbstractions
using HNSW
const KA = KernelAbstractions

# Includes
include("defines.jl")
include("macros.jl")
include("internals.jl")
include("utils.jl")
include("sparsitypattern.jl")
include("errors.jl")
include("permutations.jl")
include("input.jl")
include("kernels.jl")
include("models/VecchiaModel.jl")
include("models/legacy.jl")
include("models/api.jl")

# Exports
export VecchiaMLE_Run, VecchiaMLEInput, VecchiaModel
export recover_factor

end
