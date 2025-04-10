
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
Input to the VecchiaMLE analysis, needs to be filled out by the user!
The fields to the struct are as follows:\n

* `n`: The Square root size of the problem. I.e., the length of one side of ptGrid.  
* `k`: The "number of neighbors", number of conditioning points regarding the Vecchia Approximation. 
* `samples`: The Samples in which to generate the output. If you have no samples consult the Documentation.
* `Number_of_Samples`: The Number of Samples the user gives input to the program.
* `MadNLP_print_level`: The print level to the optimizer. Can be ignored, and will default to ERROR.
* `mode`: The opertaing mode to the analysis(GPU or CPU). The mapping is [1: 'CPU', 2: 'GPU'].
* `KKT_System`: Use the default Sparse KKT System or the custom VecchiaKKTSystem. The mapping is [1: 'Sparse System', 2: 'VecchiaKKTSystem'].

"""
mutable struct VecchiaMLEInput{M}
    n::Int
    k::Int
    samples::M
    Number_of_Samples::Int
    MadNLP_print_level::Int
    mode::Int
    KKT_System::Int

    function VecchiaMLEInput(n::Int, k::Int, samples::M, Number_of_Samples::Int, MadNLP_print_Level::Int=5, mode::Int=1, KKT_System::Int=1) where {M<:AbstractMatrix}
        return new{M}(n, k, samples, Number_of_Samples, MadNLP_print_Level, mode, KKT_System)
    end
end
