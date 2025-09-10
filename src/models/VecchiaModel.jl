function VecchiaModelMeta(::Type{S}, cache::VecchiaCache, x0::S) where {S<:AbstractArray}
    T = eltype(S)

    # TODO: Update ncon to 3n for the box diagonal constraints
    nvar::Int = cache.nnzL + cache.n
    ncon::Int = cache.n 

    # Allocating data    
    y0::S   = fill!(S(undef, ncon), zero(T))
    lcon::S = fill!(S(undef, ncon), zero(T))
    ucon::S = fill!(S(undef, ncon), zero(T))    
    lvar::S = fill!(S(undef, nvar), -Inf)
    uvar::S = fill!(S(undef, nvar),  Inf)

    # Number of nonzeros in the the lower triangular part of the Hessians
    
    return NLPModelMeta{T, S}(
        nvar,
        ncon = ncon,
        x0 = x0,
        name = "Vecchia_manual",
        nnzj = 2*cache.n,
        nnzh = cache.nnzh_tri_lag,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        lvar = lvar,
        uvar = uvar,
        minimize=true,
        islp=false,
        lin_nnzj = 0
    )
    
end

# Constructing the vecchia cache used everywhere in the code below.
function VecchiaModelCache(::Type{S}, 
    samples::AbstractMatrix, lambda::Real, rowsL::Vector{Int}, colptrL::Vector{Int},
    lvar_diag::Vector{V}, uvar_diag::Vector{V},
    arch::Symbol
)::VecchiaCache where {S <: AbstractVector, V <: AbstractFloat}
    
    M::Int = size(samples, 1)
    n::Int = length(colptrL) - 1
    T = eltype(S)

    nnzL::Int = length(rowsL)
    m = diff(colptrL)
    
    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj::Int = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag::Int = nnzh_tri_obj + n

    # Check for architecture. 
    # NOTE: Only valid since there is only 1 gpu architecture instanced.
    if S != Vector{Float64}

        offsets = CuVector{Int}(cumsum([0; m[1:end-1]]))  
        B = [CuMatrix{T}(undef, 0, 0)]
        rowsL = CuVector{Int}(rowsL)
        colptrL = CuVector{Int}(colptrL)
        m = CuVector{Int}(m)
        lvar_diag = CuVector{V}(lvar_diag)
        uvar_diag = CuVector{V}(uvar_diag)
    else
        offsets = Int[]
        B = [Matrix{T}(undef, m[j], m[j]) for j = 1:n]
    end

    hess_obj_vals::S = S(undef, nnzh_tri_obj)

    vecchia_build_B!(B, samples, lambda, rowsL, colptrL, hess_obj_vals, n, m)

    diagL = view(colptrL, 1:n)
    buffer::S = S(undef, nnzL)

    return VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1]), typeof(lvar_diag), typeof(uvar_diag)}(
        n, M, nnzL, colptrL, rowsL, diagL, 
        m, offsets, B, nnzh_tri_obj, nnzh_tri_lag, hess_obj_vals, 
        buffer, lvar_diag, uvar_diag, arch
    )
end

# Only two modes instantiated!!
VecchiaModelMetaCPU(samples::Matrix{T}, cache::VecchiaCache, x0::AbstractVector) where {T <: AbstractFloat} = VecchiaModelMeta(Vector{Float64}, cache, x0)
VecchiaModelMetaGPU(samples::CuMatrix{Float64, B}, cache::VecchiaCache, x0::AbstractVector) where {B} = VecchiaModelMeta(CuVector{Float64,B}, cache, x0)

VecchiaModelCacheCPU(samples::Matrix{T}, lambda::Real, 
    rowsL::AbstractVector, colptrL::AbstractVector, 
    lvar_diag::AbstractVector, uvar_diag::AbstractVector,
    arch::Symbol
) where {T <: AbstractFloat} = VecchiaModelCache(Vector{Float64}, samples, lambda, rowsL, colptrL, lvar_diag, uvar_diag, arch)

VecchiaModelCacheGPU(samples::CuMatrix{Float64, B}, lambda::Real, 
    rowsL::AbstractVector, colptrL::AbstractVector, 
    lvar_diag::AbstractVector, uvar_diag::AbstractVector,
    arch::Symbol
) where {B} = VecchiaModelCache(CuVector{Float64, B}, samples, lambda, rowsL, colptrL, lvar_diag, uvar_diag, arch)


######################################################################################################
#       NLPModels Functions
######################################################################################################


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
