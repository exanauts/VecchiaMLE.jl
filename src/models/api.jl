# The objective of the optimization problem.
function NLPModels.obj(nlp::VecchiaModel, x::AbstractVector)
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_obj)

    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    t1 = -nlp.cache.M * sum(z)

    # n is the number of blocks Bj in B
    # m is a vector of length n that gives the dimensions of each block Bj
    vecchia_mul!(nlp.cache.buffer, nlp.cache.B, nlp.cache.hess_obj_vals, x, nlp.cache.n, nlp.cache.m, nlp.cache.offsets)
    y = view(x, 1:nlp.cache.nnzL)
    t2 = dot(nlp.cache.buffer, y)

    return t1 + 0.5 * t2
end

# The Gradient of the objective.
function NLPModels.grad!(nlp::VecchiaModel, x::AbstractVector, gx::AbstractVector)
    @lencheck nlp.meta.nvar x gx
    increment!(nlp, :neval_grad)
    
    # n is the number of blocks Bj in B
    # m is a vector of length n that gives the dimensions of each block Bj
    vecchia_mul!(gx, nlp.cache.B, nlp.cache.hess_obj_vals, x, nlp.cache.n, nlp.cache.m, nlp.cache.offsets)
    gx_z = view(gx, nlp.cache.nnzL+1:nlp.meta.nvar)
    gx_z .= .- nlp.cache.M
    return gx
end

# Details the hessian structure. 
# All work is delegated to vecchia_generate_hess_tri_structure!
function NLPModels.hess_structure!(nlp::VecchiaModel, hrows::AbstractVector, hcols::AbstractVector)
    @lencheck nlp.meta.nnzh hrows 
    @lencheck nlp.meta.nnzh hcols

    # stored as lower triangular!
    vecchia_generate_hess_tri_structure!(nlp.meta.nnzh, nlp.cache.n, nlp.cache.m, hrows, hcols)
    return hrows, hcols
end

# Fills out Hessian non zeros.
function NLPModels.hess_coord!(nlp::VecchiaModel, x::AbstractVector, hvals::AbstractVector; obj_weight::Real=1.0)
    @lencheck nlp.meta.nnzh hvals
    increment!(nlp, :neval_hess)
    
    view(hvals, 1:nlp.cache.nnzh_tri_obj) .= nlp.cache.hess_obj_vals .* obj_weight
    return hvals
end

function NLPModels.hess_coord!(nlp::VecchiaModel, x::AbstractVector, y::AbstractVector, hvals::AbstractVector; obj_weight::Real=1.0)
    @lencheck nlp.meta.nnzh hvals
    increment!(nlp, :neval_hess)

    view(hvals, 1:nlp.cache.nnzh_tri_obj) .= nlp.cache.hess_obj_vals .* obj_weight
    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    view(hvals, nlp.cache.nnzh_tri_obj+1:nlp.cache.nnzh_tri_lag) .= y .* exp.(z)
    return hvals
end

# Hessian - vector product
# Not needed in current iteration of optimization, 
# but might be useful in the future.
function NLPModels.hprod!(nlp::VecchiaModel, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real=1.0)
    @lencheck nlp.meta.nvar v 
    @lencheck nlp.meta.nvar Hv
    increment!(nlp, :neval_hprod)
    
    # n is the number of blocks Bj in B
    # m is a vector of length n that gives the dimensions of each block Bj
    vecchia_mul!(Hv, nlp.cache.B, nlp.cache.hess_obj_vals, v, nlp.cache.n, nlp.cache.m, nlp.cache.offsets)
    view(Hv, 1:nlp.cache.nnzL) .*= obj_weight
    return Hv
end

function NLPModels.hprod!(nlp::VecchiaModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real=1.0)
    @lencheck nlp.meta.nvar v 
    @lencheck nlp.meta.nvar Hv
    increment!(nlp, :neval_hprod)
    
    # n is the number of blocks Bj in B
    # m is a vector of length n that gives the dimensions of each block Bj
    vecchia_mul!(Hv, nlp.cache.B, nlp.cache.hess_obj_vals, v, nlp.cache.n, nlp.cache.m, nlp.cache.offsets)
    view(Hv, 1:nlp.cache.nnzL) .*= obj_weight

    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    v_z = view(v, nlp.cache.nnzL+1:nlp.meta.nvar)
    view(Hv, nlp.cache.nnzL+1:nlp.meta.nvar) .= y .* exp.(z) .* v_z
    return Hv
end

# Fills out the constraint values.
function NLPModels.cons!(nlp::VecchiaModel, x::AbstractVector, c::AbstractVector)
    @lencheck nlp.cache.n c
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_cons)

    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    c .= exp.(z) .- view(x, nlp.cache.diagL)
    return c
end

# Lays out the Jacobian structure.
function NLPModels.jac_structure!(nlp::VecchiaModel, jrows::AbstractVector, jcols::AbstractVector)
    @lencheck 2*nlp.cache.n jrows
    @lencheck 2*nlp.cache.n jcols

    copyto!(view(jcols, 1:nlp.cache.n), view(nlp.cache.colptrL, 1:nlp.cache.n))
    view(jcols, (1:nlp.cache.n).+nlp.cache.n) .= (1:nlp.cache.n).+nlp.cache.nnzL
    view(jrows, 1:nlp.cache.n) .= 1:nlp.cache.n
    view(jrows, (1:nlp.cache.n).+nlp.cache.n) .= 1:nlp.cache.n
    return jrows, jcols
end

# Specifies values of the Jacobian. 
# First n values is -1 (for diagonal entries)
# Second n values are the constraint gradients.
function NLPModels.jac_coord!(nlp::VecchiaModel, x::AbstractVector, jvals::AbstractVector)
    @lencheck 2*nlp.cache.n jvals
    increment!(nlp, :neval_jac)

    fill!(view(jvals, 1:nlp.cache.n), -1.0)
    view(jvals, nlp.cache.n+1:nlp.meta.nnzj) .= exp.(view(x, nlp.cache.nnzL+1:nlp.meta.nvar))
    return jvals
end

# Jacobian - vector product
# Not needed in current iteration of optimization, 
# but might be useful in the future.
function NLPModels.jprod!(nlp::VecchiaModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)    
    @lencheck nlp.cache.n Jv 
    @lencheck nlp.cache.n+nlp.cache.nnzL v x
    increment!(nlp, :neval_jprod)

    fill!(Jv, 0.0)
    copyto!(
        Jv, 1,
        -view(v, view(nlp.cache.colptrL, 1:nlp.cache.n)) 
            .+ exp.(view(x, nlp.cache.nnzL+1:nlp.meta.nvar)) 
            .* view(v, (1:nlp.cache.n).+nlp.cache.nnzL),
        1, nlp.cache.n
    )
    return Jv
end

# Jacobian Transpose - vector product.
# Not needed in current iteration of optimization, 
# but might be useful in the future.
function NLPModels.jtprod!(nlp::VecchiaModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    # (nnzL + n) x n
    @lencheck nlp.cache.n+nlp.cache.nnzL Jtv 
    @lencheck nlp.cache.n+nlp.cache.nnzL x
    @lencheck nlp.cache.n v
    increment!(nlp, :neval_jtprod)

    fill!(Jtv, 0.0)
    copyto!(view(Jtv, view(nlp.cache.colptrL, 1:nlp.cache.n)), 1, -v, 1, nlp.cache.n)
    copyto!(Jtv, nlp.cache.nnzL+1, exp.(view(x, nlp.cache.nnzL+1:nlp.meta.nvar)) .* v, 1, nlp.cache.n)
    return Jtv
end
