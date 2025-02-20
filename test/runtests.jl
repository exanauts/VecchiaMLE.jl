using VecchiaMLE
using Test
using JuMP
using Ipopt, NLPModelsJuMP
using MadNLP, MadNLPGPU
using SparseArrays
using LinearAlgebra
using CUDA

using .VecchiaMLE
using VecchiaMLE: COMPUTE_MODE_CPU, COMPUTE_MODE_GPU

include("models/JumpModel.jl")
include("models/VecchiaMLE_models.jl")
include("test_cpu_compatible_with_jump.jl")
include("test_cpu_diagnostics.jl")

if CUDA.has_cuda()
    include("test_gpu_compatible_with_jump.jl")
    include("test_gpu_diagnostics.jl")
    include("test_cpu_compatible_with_gpu.jl")
end

include("test_model_inputs.jl")
