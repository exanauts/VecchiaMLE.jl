module VecchiaMLECUDAExt

using LinearAlgebra
using NLPModels
using VecchiaMLE
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions

function VecchiaMLE.VecchiaModel(I::Vector{Int}, J::Vector{Int}, samples::CuMatrix{T};
                                 lvar_diag::Union{Nothing,CuVector{T}}=nothing, uvar_diag::Union{Nothing,CuVector{T}}=nothing,
                                 lambda::Real=0, format::Symbol=:coo, uplo::Symbol=:L) where T
    S = CuArray{T, 1, CUDA.DeviceMemory}
    cache = VecchiaMLE.create_vecchia_cache(I, J, samples, T(lambda), format, uplo)

    nvar = length(cache.rowsL) + length(cache.colptrL) - 1
    ncon = length(cache.colptrL) - 1

    # Allocating data
    x0 = fill!(S(undef, nvar), zero(T))
    y0 = fill!(S(undef, ncon), zero(T))
    lcon = fill!(S(undef, ncon), zero(T))
    ucon = fill!(S(undef, ncon), zero(T))
    lvar = fill!(S(undef, nvar), -Inf)
    uvar = fill!(S(undef, nvar),  Inf)

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

    view(x0, cache.diagL) .= 1.0

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


function VecchiaMLE.create_vecchia_cache(I::Vector{Int}, J::Vector{Int}, samples::CuMatrix{T},
                                         lambda::T, format::Symbol, uplo::Symbol) where {T}
    S = CuArray{T, 1, CUDA.DeviceMemory}
    Msamples, n = size(samples)

    if format == :coo
        nnz_coo = length(I)
        V = ones(Int, nnz_coo)
        P = sparse(I, J, V, n, n)

        # SPARSITY PATTERN OF L IN CSC FORMAT.
        rowsL = P.rowval
        colptrL = P.colptr
    elseif format == :csc
        rowsL = I
        colptrL = J
    else
        error("Unsupported format = $format for the sparsity pattern.")
    end

    nnzL = length(rowsL)
    m = [colptrL[j+1] - colptrL[j] for j in 1:n]

    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag = nnzh_tri_obj + n

    offsets = cumsum([0; m[1:end-1]]) |> CuVector{Int}
    B = [CuMatrix{T}(undef, 0, 0)]

    rowsL = CuVector{Int}(rowsL)
    colptrL = CuVector{Int}(colptrL)
    m = CuVector{Int}(m)

    hess_obj_vals = S(undef, nnzh_tri_obj)
    VecchiaMLE.vecchia_build_B!(B, samples, lambda, rowsL, colptrL, hess_obj_vals, n, m)

    if uplo == :L
        diagL = colptrL[1:n]
    elseif uplo == :U
        diagL = colptrL[2:n+1]
        diagL .-= 1
    else
        error("Unsupported uplo = $uplo")
    end
    buffer = S(undef, nnzL)

    return VecchiaMLE.VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1])}(
        n, Msamples, nnzL,
        colptrL, rowsL, diagL,
        m, offsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end

function VecchiaMLE.recover_factor(nlp::VecchiaModel{T,<:CuVector{T}}, solution::CuVector{T}) where T
    n = nlp.cache.n
    colptr = nlp.cache.colptrL
    rowval = nlp.cache.rowsL
    nnz_factor = length(rowval)
    nzval = solution[1:nnz_factor]
    factor = CuSparseMatrixCSC(colptr, rowval, nzval, (n, n))
    return factor
end

function VecchiaMLE.vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, hess_obj_vals::CuVector{T},
                                 x::CuVector{T}, n::Int, m::CuVector{Int}, offsets::CuVector{Int}) where T <: AbstractFloat
    # Reset the vector y
    fill!(y, zero(T))

    # Launch the kernel
    backend = KernelAbstractions.get_backend(y)
    kernel = vecchia_mul_kernel!(backend)
    kernel(y, hess_obj_vals, x, m, offsets, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return y
end

function VecchiaMLE.vecchia_build_B!(B::Vector{<:CuMatrix{T}}, samples::CuMatrix{T}, lambda::T,
	                                 rowsL::CuVector{Int}, colptrL::CuVector{Int}, hess_obj_vals::CuVector{T},
	                                 n::Int, m::CuVector{Int}) where T <: AbstractFloat
    # Launch the kernel
    backend = KernelAbstractions.get_backend(samples)
    r = size(samples, 1)
    kernel = vecchia_build_B_kernel!(backend)
    kernel(hess_obj_vals, samples, lambda, rowsL, colptrL, m, r, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function VecchiaMLE.vecchia_generate_hess_tri_structure!(nnzh::Int, n::Int, colptr_diff::CuVector{Int}, 
                                                         hrows::CuVector{Int}, hcols::CuVector{Int})
    # reset hrows, hcols
    fill!(hrows, one(Int))
    fill!(hcols, one(Int))

    # launch the kernel
    backend = KernelAbstractions.get_backend(hrows)
    kernel = vecchia_generate_hess_tri_structure_kernel!(backend)

    f(x) = (x * (x+1)) ÷ 2

    # NOTE: Might be a race condition here. Solution is to store them in the first indices of each thread. 
    view(hrows, 2:n) .+= cumsum(f.(view(colptr_diff, 1:n-1)))
    view(hcols, 2:n) .+= cumsum(view(colptr_diff, 1:n-1))

    kernel(nnzh, n, colptr_diff, view(hrows, 1:n), view(hcols, 1:n), hrows, hcols, ndrange = n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function vecchia_mul_kernel!(y, @Const(hess_obj_vals), @Const(x), @Const(m), @Const(offsets))
    index = @index(Global)
    offset = offsets[index]
    mj = m[index]
    pos2 = 0
    for i = 1:index-1
        pos2 += m[i] * (m[i] + 1) ÷ 2
    end

    # Perform the matrix-vector multiplication for the current symmetric block
    for j in 1:mj
        idx1 = (j - 1) * (mj + 1) - j * (j-1) ÷ 2
        for i in j:mj
            idx2 = idx1 + (i - j + 1)
            val = hess_obj_vals[pos2+idx2]

            # Diagonal element contributes only once
            if i == j
                y[offset+i] += val * x[offset+j]
            else
                y[offset+i] += val * x[offset+j]
                y[offset+j] += val * x[offset+i]  # due to symmetry A[i,j] = A[j,i]
            end
        end
    end
    nothing
end


@kernel function vecchia_build_B_kernel!(hess_obj_vals, @Const(samples), @Const(lambda), @Const(rowsL), @Const(colptrL), @Const(m), @Const(r))
    index = @index(Global)
    pos = colptrL[index]
    mj = m[index]
    pos2 = 0
    for i = 1:index-1
        pos2 += m[i] * (m[i] + 1) ÷ 2
    end

    k = 0
    for s in 1:mj
        for t in s:mj
            acc = 0.0
            for i = 1:r
                acc += samples[i, rowsL[pos+t-1]] * samples[i, rowsL[pos+s-1]]
            end
            k = k + 1
            hess_obj_vals[pos2+k] = acc
            if (lambda != 0) && (s == t) && (s != 1)
                hess_obj_vals[pos2+k] += lambda
            end
        end
    end
    nothing
end


@kernel function vecchia_generate_hess_tri_structure_kernel!(
    @Const(nnzh), @Const(n), @Const(colptr_diff), @Const(carry_offsets), @Const(idx_offsets),
    hrows, hcols
)
    thread_idx = @index(Global) # in 1:n
    m = colptr_diff[thread_idx]
    carry = carry_offsets[thread_idx]
    idx = idx_offsets[thread_idx]

    for j in 1:m
        for k in carry:m - j + carry
            hrows[k] = (idx + j - 1) + (k - carry)
            hcols[k] = idx + j - 1
        end
        carry += m - j + 1
    end

    # fill one index of the tail for each thread
    @inbounds hrows[nnzh-n + thread_idx] = hrows[nnzh-n] + thread_idx
    @inbounds hcols[nnzh-n + thread_idx] = hrows[nnzh-n] + thread_idx
end

end
