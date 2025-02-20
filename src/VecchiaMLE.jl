module VecchiaMLE

# External Packages
using NLPModels, LinearAlgebra
using AdaptiveKDTrees, AdaptiveKDTrees.KNN
using MadNLP, MadNLPGPU
using SparseArrays 
using CUDA
using BesselK, SpecialFunctions
using Distributions, Statistics

# Includes
include("VecchiaMLE_defines.jl")
include("VecchiaMLE_utils.jl")
include("VecchiaMLE_input.jl")
include("VecchiaMLE_kernels.jl")
include("models/VecchiaMLE_NLPModel.jl")

# Exports
export VecchiaMLE_Run, VecchiaMLEInput
end
