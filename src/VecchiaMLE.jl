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
include("VecchiaMLE_defines.jl")
include("VecchiaMLE_utils.jl")
include("VecchiaMLE_input.jl")
include("VecchiaMLE_kernels.jl")
include("models/VecchiaMLE_NLPModel.jl")

# Exports
export VecchiaMLE_Run, VecchiaMLEInput
end
