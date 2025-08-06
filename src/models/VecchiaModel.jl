function VecchiaModel(::Type{S}, iVecchiaMLE::VecchiaMLEInput) where {S<:AbstractArray}
    T = eltype(S)

    cache::VecchiaCache = create_vecchia_cache(S, iVecchiaMLE)
    nvar::Int = length(cache.rowsL) + length(cache.colptrL) - 1

    # calculate nnzh
    ncon::Int = length(cache.colptrL) - 1
    x0_::S = fill!(S(undef, nvar), zero(T))
    # check x0
    if !isnothing(iVecchiaMLE.x0) 
        if mapreduce(x -> x > 0, &, view(iVecchiaMLE.x0, cache.diagL))       
            view(x0_, 1:cache.nnzL) .= iVecchiaMLE.x0
            view(x0_, (1:cache.n).+cache.nnzL) .= log.(view(iVecchiaMLE.x0, cache.diagL))
        else
            @warn "User given x0 is not feasible. Setting x0 such that the initial Vecchia approximation is the identity."
            view(x0_, cache.diagL, one(T))
        end
    end 

    y0::S = fill!(S(undef, ncon), zero(T))
    lcon::S = fill!(S(undef, ncon), zero(T))
    ucon::S = fill!(S(undef, ncon), zero(T))    
    lvar::S = fill!(S(undef, nvar), -Inf)
    uvar::S = fill!(S(undef, nvar), Inf)

    #TODO: Add checks to sanitize_input, also provide bounds if none were given in VecchiaMLEInput constructor
    if !isnothing(iVecchiaMLE.lvar_diag)
        view(lvar, cache.diagL) .= iVecchiaMLE.lvar_diag
        view(uvar, cache.nnzL+1:nvar) .= log.(iVecchiaMLE.lvar_diag)
    end

    if !isnothing(iVecchiaMLE.uvar_diag)
        view(uvar, cache.diagL) .= iVecchiaMLE.uvar_diag
        view(uvar, cache.nnzL+1:nvar) .= log.(iVecchiaMLE.uvar_diag)
    end
    #TODO: End TODO

    meta = NLPModelMeta{T, S}(
        nvar,
        ncon = ncon,
        x0 = x0_,
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
    
    return VecchiaModel(meta, Counters(), cache)
end

# Only two modes instantiated!!
VecchiaModelCPU(samples::Matrix{T}, iVecchiaMLE::VecchiaMLEInput) where {T <: AbstractFloat} = VecchiaModel(Vector{Float64}, iVecchiaMLE)
VecchiaModelGPU(samples::CuMatrix{Float64, B}, iVecchiaMLE::VecchiaMLEInput) where {B} = VecchiaModel(CuVector{Float64,B}, iVecchiaMLE)

# Constructing the vecchia cache used everywhere in the code below.
function create_vecchia_cache(::Type{S}, iVecchiaMLE::VecchiaMLEInput)::VecchiaCache where {S <: AbstractVector}
    Msamples::Int = size(iVecchiaMLE.samples, 1)
    Lsamples::Int = size(iVecchiaMLE.samples, 2)
    n::Int = iVecchiaMLE.n
    T = eltype(S)

    # SPARSITY PATTERN OF L IN CSC FORMAT.
    if !isnothing(iVecchiaMLE.rowsL)
        rowsL, colsL, colptrL = iVecchiaMLE.rowsL, iVecchiaMLE.colsL, iVecchiaMLE.colptrL
    else
        rowsL, colsL, colptrL = sparsitypattern(iVecchiaMLE.ptset, iVecchiaMLE.k, iVecchiaMLE.metric, iVecchiaMLE.sparsitygen)
    end

    nnzL::Int = length(rowsL)
    m = [colptrL[j+1] - colptrL[j] for j in 1:n]

    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj::Int = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag::Int = nnzh_tri_obj + n

    if S != Vector{Float64}

        offsets = cumsum([0; m[1:end-1]]) |> CuVector{Int}
        B = [CuMatrix{T}(undef, 0, 0)]
        rowsL = CuVector{Int}(rowsL)
        colsL = CuVector{Int}(colsL)
        colptrL = CuVector{Int}(colptrL)
        m = CuVector{Int}(m)
    else
        offsets = Int[]
        B = [Matrix{T}(undef, m[j], m[j]) for j = 1:n]
    end
    hess_obj_vals::S = S(undef, nnzh_tri_obj)

    vecchia_build_B!(B, iVecchiaMLE.samples, iVecchiaMLE.lambda, rowsL, colptrL, hess_obj_vals, n, m)

    diagL = view(colptrL, 1:n)
    buffer::S = S(undef, nnzL)

    return VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1])}(
        n, Msamples, nnzL,
        colptrL, rowsL, colsL, diagL,
        m, offsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
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
