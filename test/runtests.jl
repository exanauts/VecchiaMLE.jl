using VecchiaMLE
using Test
using JuMP
using Ipopt, NLPModelsJuMP
using MadNLP, MadNLPGPU
using SparseArrays
using LinearAlgebra
using CUDA
using NLPModels, NLPModelsTest
# using Random

using .VecchiaMLE
using VecchiaMLE: CPU, GPU

include("models/JumpModel.jl")
include("models/VecchiaMLE_models.jl")

include("test_cpu_compatible_with_jump.jl")
include("test_cpu_diagnostics.jl")
include("test_memory_allocation_cpu.jl")
include("test_custom_kkt_cpu.jl")
include("test_kkt_system_comparable_cpu.jl")
include("test_model_inputs.jl")
include("test_abnormal_ptGrid.jl")

if CUDA.has_cuda()
    include("test_gpu_compatible_with_jump.jl")
    include("test_gpu_diagnostics.jl")
    include("test_cpu_compatible_with_gpu.jl")
    include("test_memory_allocation_gpu.jl")
    include("test_custom_kkt_gpu.jl")
    include("test_kkt_system_comparable_gpu.jl")
end
