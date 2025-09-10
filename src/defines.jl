
"""
Computer architecture on which the analysis will be run. Currently
only defined as cpu or gpu, but in the future could encompass
different gpu architectures. 

## Supported Architectures
- `cpu` (`:cpu`) : CPU mode. 
- `gpu` (`:gpu`) : GPU mode. Currently only CUDA is supported. 
"""
const ARCHITECTURES = (:cpu, :gpu)

"""
Print level of the program. Describes the verbosity of the package. Currently
only implemented at the solver level.

## Supported Print Levels
- `TRACE` (`:VTRACE`) : Most Verbose.
- `DEBUG` (`:VDEBUG`) : Only useful for debugging.
- `INFO`  (`:VINFO`)  : Reports minor discrepancies in the code.
- `WARN`  (`:VWARN`)  : Reports major discrepancies in the code.
- `ERROR` (`:VERROR`) : Reports errors that prevent normal execution.
- `FATAL` (`:VFATAL`) : Catastrophic failure. 
"""
const PRINT_LEVEL = (:VTRACE, :VDEBUG, :VINFO, :VWARN, :VERROR, :VFATAL)

"""
Supported solvers for the optimization problem.
    
## Supported solvers
- `MadNLP` (`:madnlp`) : https://github.com/MadNLP/MadNLP.jl
- `Ipopt`  (`:ipopt`)  : https://github.com/jump-dev/Ipopt.jl 
- `KNITRO` (`:knitro`) : https://github.com/jump-dev/KNITRO.jl
"""
const SUPPORTED_SOLVERS = (:madnlp, :ipopt, :knitro)

"""
Specification for the Sparsity Pattern generation algorithm. 

## Supported Sparsity Patterns
- `NearestNeighbors` (`:NN`) : Based on https://github.com/KristofferC/NearestNeighbors.jl
- `HNSW` (`:HNSW`) : Based on https://github.com/JuliaNeighbors/HNSW.jl
- `Custom` (`:USERGIVEN`) : Given by user. You don't need to fill this field out, but the given sparsity pattern must be in CSC format. 
"""
const SPARSITY_GEN = (:NN, :HNSW, :USERGIVEN)

"""
Internal struct from which to fetch persisting objects in the optimization function.
There is no need for a user to mess with this!

## Fields
- `n` : Size of the problem
- `M` : Number of Samples
- `nnzL` : Number of nonzeros in L
- `colptrL` : Array which indexes the beginning of each column in L
- `rowsL` : Row index of nonzero entries in L
- `diagL` : Position of the diagonal coefficient of L
- `m` : Number of nonzeros in each column of L
- `offsets` : Number of nonzeros in hess_obj_vals before the block Bⱼ
- `B` : Vector of matrices Bⱼ, the constant blocks in the Hessian
- `nnzh_tri_obj` : Number of nonzeros in the lower triangular part of the Hessian of the objective
- `nnzh_tri_lag` : Number of nonzeros in the lower triangular part of the Hessian of the Lagrangian
- `hess_obj_vals` : Nonzeros of the lower triangular part of the Hessian of the objective
- `buffer` : Additional buffer needed for the objective
- `lvar_diag` : The lower bounds for the diagonal of L
- `uvar_diag` : The uuper bounds for the diagonal of L
"""
struct VecchiaCache{T, S, VI, M1, Vl, Vu}
    n::Int                              # Size of the problem
    M::Int                              # Number of Samples
    nnzL::Int                           # Number of nonzeros in L
    colptrL::VI                         # Array which indexes the beginning of each column in L
    rowsL::VI                           # Row index of nonzero entries in L
    diagL::VI                           # Position of the diagonal coefficient of L
    m::VI                               # Number of nonzeros in each column of L
    offsets::VI                         # Number of nonzeros in hess_obj_vals before the block Bⱼ
    B::Vector{M1}                        # Vector of matrices Bⱼ, the constant blocks in the Hessian
    nnzh_tri_obj::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the objective
    nnzh_tri_lag::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the Lagrangian
    hess_obj_vals::S                    # Nonzeros of the lower triangular part of the Hessian of the objective
    buffer::S                           # Additional buffer needed for the objective
    lvar_diag::Vl                       # The lower bounds for diagonal TODO: Implement
    uvar_diag::Vu                       # The upper bounds for diagonal TODO: Implement
    arch::Symbol
end

"""
The Diagnostics struct that records some internals of the program that would be otherwise difficult to
recover. The fields to the struct are as follows:\n

* `create_model_time`: Time taken to create Vecchia Cache and solver init.              
* `linalg_solve_time`: Time in LinearAlgebra routines in solver.
* `solve_model_time`: Time taken to solve model in solver.
* `objective_value`: Optimal Objective value. 
* `normed_constraint_value`: Optimal norm of constraint vector.
* `normed_grad_value`: Optimal norm of gradient vector.
* `iterations`: Iterations for solver to reach optimal.
* `arch`: Architecture. See ARCHITECTURES. 
"""
mutable struct Diagnostics
    create_model_time::Float64          # Time taken to create Vecchia Cache and MadNLP init.              
    linalg_solve_time::Float64          # Time in LinearAlgebra routines in MadNLP
    solve_model_time::Float64           # Time taken to solve model in MadNLP
    objective_value::Float64            # Optimal Objective value. 
    normed_constraint_value::Float64    # Optimal norm of constraint vector.
    normed_grad_value::Float64          # Optimal norm of gradient vector.
    iterations::Int                     # Iterations for MadNLP to reach optimal.
    arch::Symbol                        # Architecture: :cpu or :gpu
end

"""
Struct needed for NLPModels. 
See `VecchiaCache` for more details
"""
mutable struct VecchiaModel{T, S, VI, M} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    cache::VecchiaCache{T, S, VI, M}
end
