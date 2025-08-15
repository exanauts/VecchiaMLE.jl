
"""
Computation mode for which the analysis to run.
Generally, we should see better performance at higher n values for the gpu.
"""
@enum ComputeMode cpu=1 gpu=2

"""
Print level of the program.
Not implemented yet, but will be given by the user to
determine the print level of both VecchiaMLE and MadNLP.
"""
@enum PrintLevel VTRACE=1 VDEBUG=2 VINFO=3 VWARN=4 VERROR=5 VFATAL=6

"""
Supported solvers for the optimization problem. 
"""
const SUPPORTED_SOLVERS = Set([:madnlp, :ipopt, :knitro])

"""
Specification for the Sparsity Pattern generation algorithm. 
"""
@enum SparsityPatternGeneration NN=1 HNSW=2

"""
    Dict to map VecchiaMLE Loglevel to MadNLP LogLevel. Also for Ints. 
"""
const PRINT_LEVEL_TO_MADNLP = Dict(
    VTRACE => MadNLP.TRACE,
    VDEBUG => MadNLP.DEBUG,
    VINFO  => MadNLP.INFO,
    VWARN  => MadNLP.WARN,
    VERROR => MadNLP.ERROR,
    VFATAL => MadNLP.ERROR,
    1      => MadNLP.TRACE,
    2      => MadNLP.DEBUG,
    3      => MadNLP.INFO,
    4      => MadNLP.WARN,
    5      => MadNLP.ERROR
)

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
* `linalg_solve_time`: Time in LinearAlgebra routines in MadNLP
* `solve_model_time`: Time taken to solve model in MadNLP
* `objective_value`: Optimal Objective value. 
* `normed_constraint_value`: Optimal norm of constraint vector.
* `normed_grad_value`: Optimal norm of gradient vector.
* `iterations`: Iterations for MadNLP to reach optimal.
* `mode`: Operation mode: cpu or gpu
"""
mutable struct Diagnostics
    create_model_time::Float64          # Time taken to create Vecchia Cache and MadNLP init.              
    linalg_solve_time::Float64          # Time in LinearAlgebra routines in MadNLP
    solve_model_time::Float64           # Time taken to solve model in MadNLP
    objective_value::Float64            # Optimal Objective value. 
    normed_constraint_value::Float64    # Optimal norm of constraint vector.
    normed_grad_value::Float64          # Optimal norm of gradient vector.
    iterations::Int              # Iterations for MadNLP to reach optimal.
    mode::ComputeMode                   # Operation mode: cpu or gpu
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
Input to the VecchiaMLE analysis, needs to be filled out by the user!
The fields to the struct are as follows:\n


#### Fields

- `n::Int`: Square root size of the problem, i.e., the length of one side of `ptset`.
- `k::Int`: Number of neighbors, representing the number of conditioning points in the Vecchia Approximation.
- `samples::M`: Samples to generate the output. Each sample should match the length of the `observed_pts` vector. If no samples are available, consult the documentation.
- `number_of_samples::Int`: Number of samples provided as input to the program.
- `MadNLP_print_level::MadNLP.LogLevels`: Print level for the optimizer. Defaults to `ERROR` if ignored.
- `mode::ComputeMode`: Operating mode for the analysis. Either `gpu` or `cpu`. Defaults to `cpu`.
- `ptset::AbstractVector`: The locations from which the samples reveal their value.
- `lvar_diag::AbstractVector`: Lower bounds on the diagonal of the sparse Vecchia approximation.
- `uvar_diag::AbstractVector`: Upper bounds on the diagonal of the sparse Vecchia approximation.
- `rowsL::AbstractVector`: The sparsity pattern rows of L if the user gives one. MUST BE IN CSC FORMAT! 
- `colsL::AbstractVector`: The sparsity pattern cols of L if the user gives one. MUST BE IN CSC FORMAT!
- `colptrL::AbstractVector`: The column pointer of L if the user gives one. MUST BE IN CSC FORMAT!
- `solver::Symbol`: Optimization solver (:madnlp, :ipopt, :knitro). Defaults to `:madnlp`.
- `solver_tol::Float64`: Tolerance for the optimization solver. Defaults to `1e-8`.
- `skip_check::Bool`: Whether or not to skip the `validate_input` function.
- `metric`: The metric by which nearest neighbors are determined. Defaults to Euclidean
- `lambda`: The regularization term scalar for the ridge term `0.5 * λ‖L - diag(L)‖²` in the objective. Defaults to 0.
- `x0`: The user may give an inital condition, but it is limiting if you do not have the sparsity pattern. 
"""
mutable struct VecchiaMLEInput{M, V, V1, Vl, Vu, Vx0}
    n::Int
    k::Int
    samples::M
    number_of_samples::Int
    plevel::MadNLP.LogLevels
    mode::ComputeMode
    ptset::V
    lvar_diag::Vl
    uvar_diag::Vu
    diagnostics::Bool
    rowsL::V1
    colsL::V1
    colptrL::V1
    solver::Symbol
    solver_tol::Float64
    skip_check::Bool
    metric::Distances.Metric
    sparsitygen::SparsityPatternGeneration
    lambda::Float64
    x0::Vx0

    function VecchiaMLEInput(
        n::Int, k::Int, 
        samples::M, number_of_samples::Int, 
        plevel::PL=5, mode::CM=1;
        ptset::V=nothing,
        lvar_diag::Vl=nothing,
        uvar_diag::Vu=nothing,
        diagnostics::Bool=false,
        rowsL::V1=nothing,
        colsL::V1=nothing,
        colptrL::V1=nothing,
        solver::Symbol=:madnlp,
        solver_tol=1e-8,
        skip_check::Bool=false,
        metric::Distances.Metric=Distances.Euclidean(),
        sparsitygen::SparsityPatternGeneration=NN,
        lambda::Real=0.0,
        x0::Vx0=nothing
    ) where
        {M <:AbstractMatrix, PL <: Union{PrintLevel, Int}, CM <: Union{ComputeMode, Int}, V <: Union{Nothing, AbstractVector, AbstractMatrix},
        V1 <: Union{Nothing, AbstractVector}, Vl <: Union{Nothing, AbstractVector}, Vu <: Union{Nothing, AbstractVector}, Vx0 <: Union{Nothing, AbstractVector}}
        m = n
        if isnothing(ptset)
            ptset_ = generate_safe_xyGrid(n)
        elseif isa(ptset, AbstractMatrix)
            ptset_ = tovector(ptset)            
        else
            ptset_ = ptset
        end
        m = length(ptset_)

        return new{M, AbstractVector, V1, Vl, Vu, Vx0}(
            m,
            k,
            samples,
            number_of_samples,
            _printlevel(plevel),
            _computemode(mode),
            ptset_,
            lvar_diag,
            uvar_diag,
            diagnostics,
            rowsL,
            colsL,
            colptrL,
            solver,
            solver_tol,
            skip_check,
            metric,
            sparsitygen,
            lambda,
            x0
        )
    end
end
