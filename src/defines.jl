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
Struct needed for NLPModels. 
There is no need for the user to mess with this!
"""
mutable struct VecchiaModel{T, S, VI, M} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    cache::VecchiaCache{T, S, VI, M}
end
