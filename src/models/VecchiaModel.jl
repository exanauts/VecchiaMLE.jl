function VecchiaModel(::Type{S}, iVecchiaMLE::VecchiaMLEInput; uplo::Symbol=:L) where {S<:AbstractArray}
    T = eltype(S)

    cache::VecchiaCache = create_vecchia_cache(S, iVecchiaMLE, uplo)

    nvar::Int = length(cache.rowsL) + length(cache.colptrL) - 1
    ncon::Int = length(cache.colptrL) - 1

    # Allocating data    
    x0_::S  = fill!(S(undef, nvar), zero(T))
    y0::S   = fill!(S(undef, ncon), zero(T))
    lcon::S = fill!(S(undef, ncon), zero(T))
    ucon::S = fill!(S(undef, ncon), zero(T))    
    lvar::S = fill!(S(undef, nvar), -Inf)
    uvar::S = fill!(S(undef, nvar),  Inf)

    # Apply box constraints to the diagonal
    if !isnothing(iVecchiaMLE.lvar_diag)
        view(lvar, cache.diagL) .= iVecchiaMLE.lvar_diag
    end
    if !isnothing(iVecchiaMLE.uvar_diag)
        view(uvar, cache.diagL) .= iVecchiaMLE.uvar_diag
    end

    !iVecchiaMLE.skip_check && apply_x0!(x0_, iVecchiaMLE, cache)

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
function create_vecchia_cache(::Type{S}, iVecchiaMLE::VecchiaMLEInput, uplo::Symbol)::VecchiaCache where {S <: AbstractVector}
    Msamples::Int = size(iVecchiaMLE.samples, 1)
    n::Int = iVecchiaMLE.n
    T = eltype(S)

    # SPARSITY PATTERN OF L IN CSC FORMAT.
    rowsL, colptrL = sparsitypattern(Val(iVecchiaMLE.sparsitygen), iVecchiaMLE)
    iVecchiaMLE.rowsL .= rowsL
    iVecchiaMLE.colptrL .= colptrL

    nnzL::Int = length(rowsL)
    m = [colptrL[j+1] - colptrL[j] for j in 1:n]

    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj::Int = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag::Int = nnzh_tri_obj + n

    # Check for architecture. 
    # NOTE: Only valid since there is only 1 gpu architecture instanced.
    if S != Vector{Float64}

        offsets = cumsum([0; m[1:end-1]]) |> CuVector{Int}
        B = [CuMatrix{T}(undef, 0, 0)]
        rowsL = CuVector{Int}(rowsL)
        colptrL = CuVector{Int}(colptrL)
        m = CuVector{Int}(m)
    else
        offsets = Int[]
        B = [Matrix{T}(undef, m[j], m[j]) for j = 1:n]
    end

    hess_obj_vals::S = S(undef, nnzh_tri_obj)

    vecchia_build_B!(B, iVecchiaMLE.samples, iVecchiaMLE.lambda, rowsL, colptrL, hess_obj_vals, n, m)

    if uplo == :L
        diagL = colptrL[1:n]
    elseif uplo == :U
        diagL = colptrL[2:n+1]
        diagL .-= 1
    else
        error("Unsupported uplo = $uplo")
    end
    buffer::S = S(undef, nnzL)

    return VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1])}(
        n, Msamples, nnzL,
        colptrL, rowsL, diagL,
        m, offsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end

function VecchiaModel(I::Vector{Int}, J::Vector{Int}, samples::Matrix{T};
                      lvar_diag::Union{Nothing,Vector{T}}=nothing, uvar_diag::Union{Nothing,Vector{T}}=nothing, lambda::Real=0, uplo::Symbol=:L) where T
    S = Vector{T}
    cache::VecchiaCache = create_vecchia_cache(I, J, samples, T(lambda), uplo)

    nvar::Int = length(cache.rowsL) + length(cache.colptrL) - 1
    ncon::Int = length(cache.colptrL) - 1

    # Allocating data    
    x0::S  = fill!(S(undef, nvar), zero(T))
    y0::S   = fill!(S(undef, ncon), zero(T))
    lcon::S = fill!(S(undef, ncon), zero(T))
    ucon::S = fill!(S(undef, ncon), zero(T))    
    lvar::S = fill!(S(undef, nvar), -Inf)
    uvar::S = fill!(S(undef, nvar),  Inf)

    # Apply box constraints to the diagonal
    if !isnothing(lvar_diag)
        view(lvar, cache.diagL) .= lvar_diag
    else
        view(lvar, cache.diagL) .= 1e-10
    end

    if !isnothing(uvar_diag)
        view(uvar, cache.diagL) .= uvar_diag
    else
        view(uvar, cache.diagL) .= 1e10
    end

    meta = NLPModelMeta{T, S}(
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
    
    return VecchiaModel(meta, Counters(), cache)
end

function create_vecchia_cache(I::Vector{Int}, J::Vector{Int}, samples::Matrix{T},
                              lambda::T, uplo::Symbol)::VecchiaCache where {T}
    S = Vector{T}
    Msamples, n = size(samples)
    nnz_coo = length(I)
    V = ones(Int, nnz_coo)
    P = sparse(I, J, V, n, n)

    # SPARSITY PATTERN OF L IN CSC FORMAT.
    rowsL = P.rowval
    colptrL = P.colptr

    nnzL = length(rowsL)
    m = [colptrL[j+1] - colptrL[j] for j in 1:n]

    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj::Int = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag::Int = nnzh_tri_obj + n

    # Check for architecture. 
    # NOTE: Only valid since there is only 1 gpu architecture instanced.
    if S != Vector{T}

        offsets = cumsum([0; m[1:end-1]]) |> CuVector{Int}
        B = [CuMatrix{T}(undef, 0, 0)]
        rowsL = CuVector{Int}(rowsL)
        colptrL = CuVector{Int}(colptrL)
        m = CuVector{Int}(m)
    else
        offsets = Int[]
        B = [Matrix{T}(undef, m[j], m[j]) for j = 1:n]
    end

    hess_obj_vals::S = S(undef, nnzh_tri_obj)

    vecchia_build_B!(B, samples, lambda, rowsL, colptrL, hess_obj_vals, n, m)

    if uplo == :L
        diagL = colptrL[1:n]
    elseif uplo == :U
        diagL = colptrL[2:n+1]
        diagL .-= 1
    else
        error("Unsupported uplo = $uplo")
    end
    buffer::S = S(undef, nnzL)

    return VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1])}(
        n, Msamples, nnzL,
        colptrL, rowsL, diagL,
        m, offsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end

function recover_factor(nlp::VecchiaModel{T,Vector{T}}, solution::Vector{T}) where T
    n = nlp.cache.n
    colptr = nlp.cache.colptrL
    rowval = nlp.cache.rowsL
    nnz_factor = length(rowval)
    nzval = solution[1:nnz_factor]
    factor = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    return factor
end
