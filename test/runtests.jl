
using VecchiaMLE
using Test
using JuMP
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
using StableRNGs

function gensamples(n, nsample)
  K = [exp(-abs(j-k)/10)*(1+abs(j-k)/10) for j in 1:n, k in 1:n]
  samps_cols = cholesky(Symmetric(K)).L*randn(StableRNG(1234), n, nsample)
  permutedims(samps_cols)
end

function banded_U(n, k)
  sp = spzeros(Bool, n, n)
  for offset in 0:k
    for j in (1+offset):n
      sp[j-offset,j] = true
    end
  end
  UpperTriangular(sp)
end
banded_L(n, k) = banded_U(n, k)'

#=
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
=#
include("test_model_solver.jl")

#=
using HSL
using MadNLPHSL

if LIBHSL_isfunctional()
    include("test_linear_solver.jl")
end
=#

using Vecchia
using StaticArrays

include("vecchiajl_ext.jl")

