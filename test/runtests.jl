using VecchiaMLE
using Test
using JuMP
using Distances
using Ipopt, NLPModelsJuMP
using MadNLP, MadNLPGPU
using SparseArrays
using LinearAlgebra
using CUDA
using NLPModels, NLPModelsTest
# using Random

using .VecchiaMLE
using VecchiaMLE: cpu, gpu

include("models/JumpModel.jl")
include("models/VecchiaMLE_models.jl")

include("test_cpu_compatible_with_jump.jl")
include("test_cpu_diagnostics.jl")
include("test_memory_allocation_cpu.jl")
include("test_different_metrics.jl")

if CUDA.has_cuda()
    include("test_gpu_compatible_with_jump.jl")
    include("test_gpu_diagnostics.jl")
    include("test_cpu_compatible_with_gpu.jl")
    include("test_memory_allocation_gpu.jl")
end

include("test_model_inputs.jl")
include("test_abnormal_ptGrid.jl")

using HSL, MadNLPHSL

if LIBHSL_isfunctional()
    include("test_linear_solver.jl")
end