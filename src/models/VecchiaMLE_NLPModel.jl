# TODO: GPU is running poorly. Something being ported from CPU?

function VecchiaModel(::Type{S}, samples::AbstractMatrix, k::Int, xyGrid) where {S<:AbstractArray}
    T = eltype(S)
    cache = create_vecchia_cache(samples, k, xyGrid, S)
    nvar_ = length(cache.rowsL) + length(cache.colptrL) - 1
    
    # The initial condition is for L to be the identity. 
    x0_::S = S(undef, nvar_)
    fill!(x0_, 0.0)
    fill!(x0_[cache.colptrL[1:end-1]], one(T))

    # calculate nnzh
    ncon::Int = length(cache.colptrL) - 1

    y0::S = fill!(S(undef, ncon), zero(T))
    ucon::S = fill!(S(undef, ncon), zero(T))
    lcon::S = fill!(S(undef, ncon), zero(T))

    meta = NLPModelMeta{T, S}(
        nvar_,
        ncon = ncon,
        x0 = x0_,
        name = "Vecchia_manual",
        nnzj = 2*cache.n,
        nnzh = cache.nnzh_tri_lag,
        y0 = y0,
        ucon = ucon,
        lcon = lcon,
        minimize=true,
        islp=false,
        lin_nnzj = 0
    )
    
    return VecchiaModel(meta, Counters(), cache)
end

# Only two modes instantiated!!
VecchiaModelCPU(samples::AbstractMatrix, k::Int, xyGrid) = VecchiaModel(Vector{Float64}, samples::AbstractMatrix, k::Int, xyGrid)
VecchiaModelGPU(samples::AbstractMatrix, k::Int, xyGrid) = VecchiaModel(CuArray{Float64, 1, CUDA.DeviceMemory}, samples::AbstractMatrix, k::Int, xyGrid)

# Constructing the vecchia cache used everywhere in the code below.
function create_vecchia_cache(samples::AbstractMatrix, k::Int, xyGrid, ::Type{S}) where {S <: AbstractVector}
    M = size(samples, 1)
    n = size(samples, 2)
    T = eltype(S)

    # SPARSITY PATTERN OF L IN COO, CSC FORMAT.
    rowsL, colsL, colptrL = SparsityPattern(xyGrid, k, "CSC")

    nnzL = length(rowsL)
    m = Int[colptrL[j+1] - colptrL[j] for j in 1:n]

    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag = nnzh_tri_obj + n

    B = [Matrix{T}(undef, m[j], m[j]) for j = 1:n]
    hess_obj_vals = Vector{T}(undef, nnzh_tri_obj)

    pos = 0
    for j in 1:n
        for s in 1:m[j]
            for t in 1:m[j]
                vt = view(samples, :, rowsL[colptrL[j] + t - 1])
                vs = view(samples, :, rowsL[colptrL[j] + s - 1])
                B[j][t, s] = dot(vt, vs)

                # Lower triangular part of the block Bⱼ
                if s ≤ t
                    pos = pos + 1
                    hess_obj_vals[pos] = B[j][t, s]
                end
            end
        end
    end
    @assert pos == nnzh_tri_obj

    if S != Vector{Float64}
        B = [CuMatrix{T}(B[j]) for j = 1:n]
        hess_obj_vals = S(hess_obj_vals)
        rowsL = CuVector{Int}(rowsL)
        colsL = CuVector{Int}(colsL)
        colptrL = CuVector{Int}(colptrL)
    end
    diagL = colptrL[1:n]
    buffer = S(undef, nnzL)

    return VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B)}(
        n, M, nnzL, 
        colptrL, rowsL, colsL, diagL,
        m, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end

# Can split this up into nlp::VecchiaModel{T, <: Vector}, x::Vector, 
# and                    nlp::VecchiaModel{T, <: CuVector}, x::CuVector
# The objective of the optimization problem.
function NLPModels.obj(nlp::VecchiaModel, x::AbstractVector)
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_obj)

    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    t1 = -nlp.cache.M * sum(z)

    b = nlp.cache.buffer
    pos = 0

    # cublasGemmGroupedBatchedEx() https://docs.nvidia.com/cuda/cublas/#cublasgemmgroupedbatchedex
    for j = 1:nlp.cache.n
        Bj = nlp.cache.B[j]
        xj = view(x, pos+1:pos+nlp.cache.m[j])
        bj = view(b, pos+1:pos+nlp.cache.m[j])
        mul!(bj, Bj, xj)
        pos = pos + nlp.cache.m[j]
    end
    y = view(x, 1:nlp.cache.nnzL)
    t2 = dot(b, y)

    return t1 + 0.5 * t2
end

# The Gradient of the objective.
function NLPModels.grad!(nlp::VecchiaModel, x::AbstractVector, gx::AbstractVector)
    @lencheck nlp.meta.nvar x gx
    increment!(nlp, :neval_grad)
    
    pos = 0
    for j = 1:nlp.cache.n
        Bj = nlp.cache.B[j]
        xj = view(x, pos+1:pos+nlp.cache.m[j])
        gxj = view(gx, pos+1:pos+nlp.cache.m[j])
        mul!(gxj, Bj, xj)
        pos = pos + nlp.cache.m[j]
    end

    gx_z = view(gx, nlp.cache.nnzL+1:nlp.meta.nvar)
    fill!(gx_z, -nlp.cache.M)

    return gx
end

# Details the hessian structure. 
# All work is delegated to generate_hessian_tri_structure!
function NLPModels.hess_structure!(nlp::VecchiaModel, hrows::AbstractVector, hcols::AbstractVector)
    @lencheck nlp.meta.nnzh hrows 
    @lencheck nlp.meta.nnzh hcols

    # stored as lower triangular!
    generate_hessian_tri_structure!(nlp.meta.nnzh, nlp.cache.n, nlp.cache.m, hrows, hcols)
    return hrows, hcols
end

# Fills out Hessian non zeros.
function NLPModels.hess_coord!(nlp::VecchiaModel, x::AbstractVector, y::AbstractVector, hvals::AbstractVector; obj_weight::Real=1.0)
    @lencheck nlp.meta.nnzh hvals
    increment!(nlp, :neval_hess)
    hvals_obj = view(hvals, 1:nlp.cache.nnzh_tri_obj)
    hvals_obj .= nlp.cache.hess_obj_vals .* obj_weight

    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    hvals_con = view(hvals, nlp.cache.nnzh_tri_obj+1:nlp.cache.nnzh_tri_lag)
    hvals_con .= y .* exp.(z)
    return hvals
end

# Hessian - vector product
# Not needed in current iteration of optimization, 
# but might be useful in the future.
function NLPModels.hprod!(nlp::VecchiaModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight::Real=1.0)
    @lencheck nlp.meta.nvar v 
    @lencheck nlp.meta.nvar Hv
    increment!(nlp, :neval_hprod)
    
    pos = 0
    for j = 1:nlp.cache.n
        Bj = nlp.cache.B[j]
        vj = view(v, pos+1:pos+nlp.cache.m[j])
        Hvj = view(Hv, pos+1:pos+nlp.cache.m[j])
        mul!(Hvj, Bj, vj)
        pos = pos + nlp.cache.m[j]
    end

    view(Hv, nlp.cache.nnzL+1:nlp.meta.nvar) .-= nlp.cache.M .* y
    return Hv
end

# Fills out the constraint values.
function NLPModels.cons!(nlp::VecchiaModel, x::AbstractVector, c::AbstractVector)
    @lencheck nlp.cache.n c
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_cons)

    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    diagL = view(x, nlp.cache.diagL)
    c .= exp.(z) .- diagL
    return c
end

# Lays out the Jacobian structure.
# TODO: A little hacky. 
function NLPModels.jac_structure!(nlp::VecchiaModel, jrows::AbstractVector, jcols::AbstractVector)
    @lencheck 2*nlp.cache.n jrows
    @lencheck 2*nlp.cache.n jcols
    
    v = collect((1:nlp.cache.n) .+ nlp.cache.nnzL)
    v2 = repeat(1:nlp.cache.n, outer=2)
    VI = typeof(jrows)
    v = VI(v)
    v2 = VI(v2)
    copyto!(jcols, 1, view(nlp.cache.colptrL, 1:nlp.cache.n), 1, nlp.cache.n)
    copyto!(jcols, 1+nlp.cache.n, v, 1, nlp.cache.n)
    copyto!(jrows, 1, v2, 1, 2*nlp.cache.n)
    return jrows, jcols
end

# Specifies values of the Jacobian. 
# First n values is -1 (for diagonal entries)
# Second n values are the constraint gradients.
function NLPModels.jac_coord!(nlp::VecchiaModel, x::AbstractVector, jvals::AbstractVector)
    @lencheck 2*nlp.cache.n jvals
    increment!(nlp, :neval_jac)

    fill!(view(jvals, 1:nlp.cache.n), -1.0)
    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    view(jvals, nlp.cache.n+1:nlp.meta.nnzj) .= exp.(z)
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
    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    copyto!(
        Jv, 1,
        -view(v, 1:nlp.cache.colptrL[1:end-1]) + exp.(z) .* view(v, (1:nlp.cache.n).+nlp.cache.nnzL),
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
    z = view(x, nlp.cache.nnzL+1:nlp.meta.nvar)
    copyto!(view(Jtv, view(nlp.cache.colptrL, 1:nlp.cache.n)), 1, -v, 1, nlp.cache.n)
    copyto!(Jtv, nlp.cache.nnzL+1, exp.(z) .* v, 1, nlp.cache.n)
    return Jtv
end

# function to generate the hessian structure in CSC format. 
# Needs to be abstracted since need to hold it in NLPModel and cache, thus two calls.
function generate_hessian_tri_structure!(nnzh::Int, n::Int, colptr_diff::Vector{Int}, hrows::AbstractVector, hcols::AbstractArray)
    carry = 1
    idx = 1
    filtered_rows_entire = zeros(length(hrows))
    filtered_cols_entire = zeros(length(hcols))
    for i in 1:n
        block_side_length = colptr_diff[i]
        block_range = idx:(idx+block_side_length-1)
    
        B_i_rows = repeat(block_range, outer=block_side_length)
        B_i_cols = repeat(block_range, inner=block_side_length)
    
        lower_tri_mask = B_i_rows .>= B_i_cols
        filtered_rows = view(B_i_rows, lower_tri_mask)
        filtered_cols = view(B_i_cols, lower_tri_mask)

        # NOTE: Fine since on CPU
        @views filtered_rows_entire[(0:length(filtered_rows)-1).+carry] .= filtered_rows
        @views filtered_cols_entire[(0:length(filtered_cols)-1).+carry] .= filtered_cols
            
        carry += length(filtered_rows)
        idx += block_side_length
    end

    #Then need the diagonal tail
    idx_to = idx + nnzh - carry

    # CPU copy
    @views filtered_rows_entire[end-n+1:end] .= idx:idx_to
    @views filtered_cols_entire[end-n+1:end] .= filtered_rows_entire[end-n+1:end]

    # One set of copies to GPU. More efficient
    copyto!(hrows, 1, filtered_rows_entire, 1, nnzh)
    copyto!(hcols, 1, filtered_cols_entire, 1, nnzh)

    return hrows, hcols
end
