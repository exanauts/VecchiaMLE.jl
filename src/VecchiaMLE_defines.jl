

if !@isdefined(COMPUTE_MODE) @enum COMPUTE_MODE CPU=1 GPU=2 end 
if !@isdefined(PRINT_LEVEL) @enum PRINT_LEVEL VTRACE=1 VDEBUG=2 VINFO=3 VWARN=4 Error=5 VFATAL=6 end

if !@isdefined(ConfigManager)
    struct ConfigManager
        n::Int                              # Size of the problem
        k::Int                              # Length of conditioning points in Vecchia Approximation
        mode::COMPUTE_MODE                  # Operation Mode: GPU or CPU
        Number_of_Samples::Int              # Number of Samples from MvNormal
        MadNLP_Print_Level::Int             # Print level for MadNLP
        samples::Matrix{Float64}            # Holds samples
    end
end    

if !@isdefined(VecchiaCache)
    struct VecchiaCache{T, S, VI, SM}
        n::Int                              # Size of the problem
        M::Int                              # Number of Samples
        nnzL::Int                           # Number of nonzeros in L
        colptrL::VI                         # Array which indexes the beginning of each column in L
        rowsL::VI                           # Row index of nonzero entries in L
        colsL::VI                           # Column index of nonzero entries in L
        diagL::VI                           # Position of the diagonal coefficient of L
        m::Vector{Int}                      # Number of nonzeros in each column of L
        B::SM                               # Vector of Matrices Bⱼ, the constant blocks in the Hessian
        nnzh_tri_obj::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the objective
        nnzh_tri_lag::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the Lagrangian
        hess_obj_vals::S                    # Nonzeros of the lower triangular part of the Hessian of the objective
        buffer::S                           # Additional buffer needed for the objective
    end
end

if !@isdefined(Diagnostics)
    mutable struct Diagnostics
        create_model_time::Float64          # Time taken to create Vecchia Cache and MadNLP init.              
        LinAlg_solve_time::Float64          # Time in LinearAlgebra routines in MadNLP
        solve_model_time::Float64           # Time taken to solve model in MadNLP
        objective_value::Float64            # Optimal Objective value. 
        normed_constraint_value::Float64    # Optimal norm of constraint vector.
        normed_grad_value::Float64          # Optimal norm of gradient vector.
        MadNLP_iterations::Integer          # Iterations for MadNLP to reach optimal.
        mode::Integer                       # Operation mode: CPU or GPU
    end
end


# Vecchia Model for NLPModels package.
if !@isdefined(VecchiaModel)
    mutable struct VecchiaModel{T, S} <: AbstractNLPModel{T, S}
        meta::NLPModelMeta{T, S}
        counters::Counters
        cache::VecchiaCache
    end
end    

if !@isdefined(VecchiaMLEInput)
    mutable struct VecchiaMLEInput
        n::Integer
        k::Integer
        samples::Matrix{Float64}
        Number_of_Samples::Integer
        MadNLP_print_level::Integer
        mode::Integer
        function VecchiaMLEInput(n::Integer, k::Integer, samples::Matrix{Float64}, Number_of_Samples::Integer, MadNLP_print_Level::Integer=5, mode::Integer=1)
            return new(n, k, samples, Number_of_Samples, MadNLP_print_Level, mode)
        end
    end
end