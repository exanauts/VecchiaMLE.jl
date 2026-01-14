using VecchiaMLE
using Test
using JuMP
using Distances
using Ipopt
using NLPModelsJuMP
using NLPModelsIpopt
using UnoSolver
using MadNLP
using MadNLPGPU
using SparseArrays
using LinearAlgebra
using CUDA
using NLPModels
using NLPModelsTest

# Includes
include("models/Jump_models.jl")
include("models/VecchiaMLE_models.jl")

# Tests
include("test_output.jl")
include("test_cpu_compatible_with_jump.jl")
include("test_memory_allocation_cpu.jl")
include("test_different_metrics.jl")

if CUDA.has_cuda()
    include("test_sparsity_pattern_hnsw.jl")
    include("test_gpu_compatible_with_jump.jl")
    include("test_cpu_compatible_with_gpu.jl")
    include("test_memory_allocation_gpu.jl")
end

include("test_abnormal_ptset.jl")
include("test_model_solver.jl")

using HSL
using MadNLPHSL

if LIBHSL_isfunctional()
    include("test_linear_solver.jl")
end
