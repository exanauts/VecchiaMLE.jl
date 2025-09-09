
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

"""
struct VecchiaCache{T, S, VI, M}
    n::Int                              # Size of the problem
    M::Int                              # Number of Samples
    nnzL::Int                           # Number of nonzeros in L
    colptrL::VI                         # Array which indexes the beginning of each column in L
    rowsL::VI                           # Row index of nonzero entries in L
    diagL::VI                           # Position of the diagonal coefficient of L
    m::VI                               # Number of nonzeros in each column of L
    offsets::VI                         # Number of nonzeros in hess_obj_vals before the block Bⱼ
    B::Vector{M}                        # Vector of matrices Bⱼ, the constant blocks in the Hessian
    nnzh_tri_obj::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the objective
    nnzh_tri_lag::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the Lagrangian
    hess_obj_vals::S                    # Nonzeros of the lower triangular part of the Hessian of the objective
    buffer::S                           # Additional buffer needed for the objective
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
There is no need for the user to mess with this!
"""
mutable struct VecchiaModel{T, S, VI, M} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    cache::VecchiaCache{T, S, VI, M}
end

"""
Input to the VecchiaMLE analysis.
## Fields

- `k::Int`: Number of neighbors, representing the number of conditioning points in the Vecchia Approximation.
- `samples::M`: Samples to generate the output. Each sample should match the length of the `observed_pts` vector.
- `plevel::Symbol`: Print level for the optimizer. See PRINT_LEVEL. Defaults to `:VERROR`.
- `arch::Symbol`: Architecture for the analysis. See ARCHITECTURES. Defaults to `:cpu`.

## Keyword Arguments

* plevel::Symbol            # Print level for the optimizer. See PRINT_LEVEL. Defaults to `ERROR`.
* arch::Symbol              # Architecture for the analysis. See ARCHITECTURES. Defaults to `:cpu`.
* ptset::AbstractVector     # The locations of the analysis. May be passed as a matrix or vector of vectors.
* lvar_diag::AbstractVector # Lower bounds on the diagonal of the sparse Vecchia approximation.
* uvar_diag::AbstractVector # Upper bounds on the diagonal of the sparse Vecchia approximation.
* rowsL::AbstractVector     # The sparsity pattern rows of L if the user gives one. MUST BE IN CSC FORMAT!
* colptrL::AbstractVector   # The column pointer of L if the user gives one. MUST BE IN CSC FORMAT!
* solver::Symbol            # Optimization solver (:madnlp, :ipopt, :knitro). Defaults to `:madnlp`.
* solver_tol::Float64       # Tolerance for the optimization solver. Defaults to `1e-8`.
* skip_check::Bool          # Whether or not to skip the `validate_input` function.
* sparsityGeneration        # The method by which to generate a sparsity pattern. See SPARSITY_GEN.
* metric::Distances.metric  # The metric by which nearest neighbors are determined. Defaults to Euclidean.
* lambda::Real              # The regularization scalar for the ridge `0.5 * λ‖L - diag(L)‖²` in the objective. Defaults to 0.
* x0::AbstractVector        # The user may give an initial condition, but it is limiting if you do not have the sparsity pattern. 
"""
mutable struct VecchiaMLEInput{M, V, V1, Vl, Vu, Vx0}
    n::Int 
    k::Int 
    samples::M
    number_of_samples::Int
    plevel::Symbol
    arch::Symbol
    ptset::V
    lvar_diag::Vl
    uvar_diag::Vu
    diagnostics::Bool
    rowsL::V1
    colptrL::V1
    solver::Symbol
    solver_tol::Float64
    skip_check::Bool
    metric::Distances.Metric
    sparsitygen::Symbol
    lambda::Float64
    x0::Vx0
end

function VecchiaMLEInput(
    n::Int, k::Int, 
    samples::M, number_of_samples::Int; 
    plevel::VPL=:VERROR, 
    arch::VAR=:cpu,
    ptset::V=generate_safe_xyGrid(n),
    lvar_diag::Vl=nothing,
    uvar_diag::Vu=nothing,
    diagnostics::Bool=false,
    rowsL::V1=nothing,
    colptrL::V1=nothing,
    solver::Symbol=:madnlp,
    solver_tol::Real=1e-8,
    skip_check::Bool=false,
    metric::Distances.Metric=Distances.Euclidean(),
    sparsitygen::Symbol=:NN,
    lambda::Real=0.0,
    x0::Vx0=nothing
) where
    {
        M   <: AbstractMatrix, 
        V   <: Union{AbstractVector, AbstractMatrix},
        V1  <: Union{Nothing, AbstractVector},
        Vl  <: Union{Nothing, AbstractVector},
        Vu  <: Union{Nothing, AbstractVector},
        Vx0 <: Union{Nothing, AbstractVector},
        VPL <: Union{Symbol, Int},
        VAR <: Union{Symbol, Int}
    }
    ptset_ = resolve_ptset(n, ptset)
    n_::Int = length(ptset_)
    


    return VecchiaMLEInput{M, typeof(ptset_), Vector{Int}, Vl, Vu, Vx0}(
        n_,
        k,
        samples,
        number_of_samples,
        convert_plevel(Val(plevel)),
        convert_computemode(Val(arch)),
        ptset_,
        lvar_diag,
        uvar_diag,
        diagnostics,
        resolve_rowsL(rowsL, n_, k),
        resolve_colptrL(colptrL, n_),
        solver,
        Float64(solver_tol),
        skip_check,
        metric,
        sparsitygen,
        Float64(lambda),
        x0
    )
end

"""
Input to the VecchiaMLE analysis. Samples are expected as row vectors. 
## Fields

* `k::Int`: Number of neighbors, representing the number of conditioning points in the Vecchia Approximation.
* `samples::M`: Samples to generate the output. Each sample should match the length of the `observed_pts` vector.

## Keyword Arguments
* plevel::Symbol            # Print level for the optimizer. See PRINT_LEVEL. Defaults to `ERROR`.
* arch::Symbol              # Architecture for the analysis. See ARCHITECTURES. Defaults to `:cpu`.
* ptset::AbstractVector     # The locations of the analysis. May be passed as a matrix or vector of vectors.
* lvar_diag::AbstractVector # Lower bounds on the diagonal of the sparse Vecchia approximation.
* uvar_diag::AbstractVector # Upper bounds on the diagonal of the sparse Vecchia approximation.
* rowsL::AbstractVector     # The sparsity pattern rows of L if the user gives one. MUST BE IN CSC FORMAT!
* colptrL::AbstractVector   # The column pointer of L if the user gives one. MUST BE IN CSC FORMAT!
* solver::Symbol            # Optimization solver (:madnlp, :ipopt, :knitro). Defaults to `:madnlp`.
* solver_tol::Float64       # Tolerance for the optimization solver. Defaults to `1e-8`.
* skip_check::Bool          # Whether or not to skip the `validate_input` function.
* sparsityGeneration        # The method by which to generate a sparsity pattern. See SPARSITY_GEN.
* metric::Distances.metric  # The metric by which nearest neighbors are determined. Defaults to Euclidean.
* lambda::Real              # The regularization scalar for the ridge `0.5 * λ‖L - diag(L)‖²` in the objective. Defaults to 0.
* x0::AbstractVector        # The user may give an initial condition, but it is limiting if you do not have the sparsity pattern. 
"""
VecchiaMLEInput(k::Int, samples; kwargs...) = VecchiaMLEInput(size(samples, 2), k, samples, size(samples, 1); kwargs...)
    
