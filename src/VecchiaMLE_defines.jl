
"""
Computation mode for which the analysis to run.
Generally, we should see better performance at higher n values for the GPU.
"""
@enum COMPUTE_MODE CPU=1 GPU=2

"""
Print level of the program.
Not implemented yet, but will be given by the user to
determine the print level of both VecchiaMLE and MadNLP.
"""
@enum PRINT_LEVEL VTRACE=1 VDEBUG=2 VINFO=3 VWARN=4 Error=5 VFATAL=6


"""
At the moment, not used!
"""
struct ConfigManager{M}
    n::Int                              # Size of the problem
    k::Int                              # Length of conditioning points in Vecchia Approximation
    mode::COMPUTE_MODE                  # Operation Mode: GPU or CPU
    Number_of_Samples::Int              # Number of Samples from MvNormal
    MadNLP_Print_Level::Int             # Print level for MadNLP
    samples::M                          # Holds samples
end

"""
Internal struct from which to fetch persisting objects in the optimization function.
There is no need for the user to mess with this!
"""
struct VecchiaCache{T, S, VI, M}
    n::Int                              # Size of the problem
    M::Int                              # Number of Samples
    nnzL::Int                           # Number of nonzeros in L
    colptrL::VI                         # Array which indexes the beginning of each column in L
    rowsL::VI                           # Row index of nonzero entries in L
    colsL::VI                           # Column index of nonzero entries in L
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

* `create_model_time`: Time taken to create Vecchia Cache and MadNLP init.              
* `LinAlg_solve_time`: Time in LinearAlgebra routines in MadNLP
* `solve_model_time`: Time taken to solve model in MadNLP
* `objective_value`: Optimal Objective value. 
* `normed_constraint_value`: Optimal norm of constraint vector.
* `normed_grad_value`: Optimal norm of gradient vector.
* `MadNLP_iterations`: Iterations for MadNLP to reach optimal.
* `mode`: Operation mode: CPU or GPU
"""
mutable struct Diagnostics
    create_model_time::Float64          # Time taken to create Vecchia Cache and MadNLP init.              
    LinAlg_solve_time::Float64          # Time in LinearAlgebra routines in MadNLP
    solve_model_time::Float64           # Time taken to solve model in MadNLP
    objective_value::Float64            # Optimal Objective value. 
    normed_constraint_value::Float64    # Optimal norm of constraint vector.
    normed_grad_value::Float64          # Optimal norm of gradient vector.
    MadNLP_iterations::Int              # Iterations for MadNLP to reach optimal.
    mode::Int                           # Operation mode: CPU or GPU
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
Input to the VecchiaMLE analysis, to be filled out by the user.

# Fields
- `n::Int`: Square root size of the problem, i.e., the length of one side of `ptGrid`.
- `k::Int`: Number of neighbors, representing the number of conditioning points in the Vecchia Approximation.
- `samples::M`: Samples to generate the output. Each sample should match the length of the `observed_pts` vector. If no samples are available, consult the documentation.
- `Number_of_Samples::Int`: Number of samples provided as input to the program.
- `ptGrid::AbstractVector`: The larger gridded space in which the observed points lie.
- `observed_pts::AbstractVector`: The observed points within `ptGrid`.
- `MadNLP_print_level::Int`: Print level for the optimizer. Defaults to `ERROR` if ignored.
- `mode::Int`: Operating mode for the analysis (`1` for 'CPU', `2` for 'GPU').
"""
mutable struct VecchiaMLEInput{M}
    n::Int
    k::Int
    samples::M
    Number_of_Samples::Int
    ptGrid::AbstractVector    
    observed_pts::AbstractVector
    MadNLP_print_level::Int
    mode::Int
end

"""
Constructs a `VecchiaMLEInput` instance with specified `ptGrid` and `observed_pts`.

# Arguments
- `n::Int`: Square root size of the problem.
- `k::Int`: Number of neighbors for the Vecchia Approximation.
- `samples::M`: Samples for output generation.
- `Number_of_Samples::Int`: Number of samples provided.
- `MadNLP_print_level::Int`: (Optional) Print level for the optimizer. Defaults to `5`.
- `mode::Int`: (Optional) Operating mode (`1` for 'CPU', `2` for 'GPU'). Defaults to `1`.
- `ptGrid::AbstractVector`: (Keyword) Larger gridded space containing the observed points.
- `observed_pts::AbstractVector`: (Keyword) Observed points within `ptGrid`.
"""
function VecchiaMLEInput(n::Int, k::Int, samples::M, Number_of_Samples::Int, MadNLP_print_level::Int=5, mode::Int=1;
    ptGrid::AbstractVector=[], observed_pts::AbstractVector=[]) where {M <: AbstractMatrix}
    iVecchiaMLE = VecchiaMLEInput(n, k, samples, Number_of_Samples, ptGrid, observed_pts, MadNLP_print_level, mode)
    sanitize_input!(iVecchiaMLE)  
    return iVecchiaMLE
end

"""
Constructs a `VecchiaMLEInput` instance with specified `ptGrid` and `observed_idx_mapping`.

# Arguments
- `n::Int`: Square root size of the problem.
- `k::Int`: Number of neighbors for the Vecchia Approximation.
- `samples::M`: Samples for output generation.
- `Number_of_Samples::Int`: Number of samples provided.
- `print_level::Int`: (Optional) Print level for the optimizer. Defaults to `5`.
- `mode::Int`: (Optional) Operating mode (`1` for 'CPU', `2` for 'GPU'). Defaults to `1`.
- `ptGrid::AbstractVector`: (Keyword) Larger gridded space containing the observed points.
- `observed_idx_mapping::AbstractVector`: (Keyword) Indices mapping to observed points within `ptGrid`.
"""
function VecchiaMLEInput(n::Int, k::Int, samples::M, Number_of_Samples::Int, print_level::Int=5, mode::Int=1;
    ptGrid::AbstractVector=[], observed_idx_mapping::AbstractVector=[]) where {M <: AbstractMatrix}
    @assert isa(observed_idx_mapping, Vector{Int}) "observed_idx_mapping is not a vector of indices!"
    @assert maximum(observed_idx_mapping) <= length(ptGrid) && minimum(observed_idx_mapping) >= 1 "observed_idx_mapping has illegal indices!"
    observed_pts = ptGrid[observed_idx_mapping]
    return VecchiaMLEInput(n, k, samples, Number_of_Samples, print_level, mode; ptGrid=ptGrid, observed_pts=observed_pts)
end