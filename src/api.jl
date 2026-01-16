function vecchia_mul!(y::Vector{T}, B::Vector{Matrix{T}}, hess_obj_vals::Vector{T},
                      x::Vector{T}, n::Int, m::Vector{Int}, offsets::Vector{Int}) where T <: AbstractFloat
    pos = 0
    for j = 1:n
        Bj = B[j]
        yj = view(y, pos+1:pos+m[j])
        xj = view(x, pos+1:pos+m[j])
        mul!(yj, Bj, xj)
        pos = pos + m[j]
    end
    return y
end

function vecchia_build_B!(B::Vector{Matrix{T}}, samples::Matrix{T}, lambda::T, rowsL::Vector{Int},
                          colptrL::Vector{Int}, hess_obj_vals::Vector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
    pos = 0
    for j in 1:n
        for s in 1:m[j]
            for t in 1:m[j]
                vt = view(samples, :, rowsL[colptrL[j] + t - 1])
                vs = view(samples, :, rowsL[colptrL[j] + s - 1])
                B[j][t, s] = dot(vt, vs)
                # Ridge regularization
                if (lambda != 0) && (s == t) && (s != 1)
                    # s == 1 means that we treat the variable related to the diagonal coefficient of column j
                    # of the sparse Cholesky factor.
                    #
                    # Only update diagonal coefficient of the Hessian that are
                    # related to the variables that represent the off-diagonal terms
                    # of the sparse Cholesky factor.
                    B[j][t, s] += lambda
                end
                # Lower triangular part of the block Bⱼ
                if s ≤ t
                    pos = pos + 1
                    hess_obj_vals[pos] = B[j][t, s]
                end
            end
        end
    end

    return nothing
end

for INT in (:Int32, :Int64)
    @eval begin
        function vecchia_generate_hess_tri_structure!(nnzh::Int, n::Int, colptr_diff::Vector{Int},
                                                      hrows::Vector{$INT}, hcols::Vector{$INT})
            carry = 1
            idx = 1
            for i in 1:n
                m = colptr_diff[i]
                    for j in 1:m
                        view(hrows, (0:(m-j)).+carry) .= (j:m).+(idx-1)
                        fill!(view(hcols, carry:carry+m-j), idx + j - 1)
                        carry += m - j + 1
                    end
                idx += m
            end

            #Then need the diagonal tail
            idx_to = idx + nnzh - carry
            view(hrows, carry:nnzh) .= idx:idx_to
            view(hcols, carry:nnzh) .= idx:idx_to

            return hrows, hcols
        end
    end
end

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

    copyto!(view(jcols, 1:nlp.cache.n), nlp.cache.diagL)
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
        -view(v, nlp.cache.diagL) 
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
    copyto!(view(Jtv, nlp.cache.diagL), 1, -v, 1, nlp.cache.n)
    copyto!(Jtv, nlp.cache.nnzL+1, exp.(view(x, nlp.cache.nnzL+1:nlp.meta.nvar)) .* v, 1, nlp.cache.n)
    return Jtv
end
