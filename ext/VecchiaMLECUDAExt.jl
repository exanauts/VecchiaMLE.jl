module VecchiaMLECUDAExt

using LinearAlgebra
using NLPModels
using VecchiaMLE
using CUDA
using CUDA.CUSPARSE
import KernelAbstractions as KA

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
    backend = KA.get_backend(y)
    kernel = VecchiaMLE.vecchia_mul_kernel!(backend)
    kernel(y, hess_obj_vals, x, m, offsets, ndrange=n)
    KA.synchronize(backend)
    return y
end

function VecchiaMLE.vecchia_build_B!(B::Vector{<:CuMatrix{T}}, samples::CuMatrix{T}, lambda::T,
	                                 rowsL::CuVector{Int}, colptrL::CuVector{Int}, hess_obj_vals::CuVector{T},
	                                 n::Int, m::CuVector{Int}) where T <: AbstractFloat
    # Launch the kernel
    backend = KA.get_backend(samples)
    r = size(samples, 1)
    kernel = VecchiaMLE.vecchia_build_B_kernel!(backend)
    kernel(hess_obj_vals, samples, lambda, rowsL, colptrL, m, r, ndrange=n)
    KA.synchronize(backend)
    return nothing
end

function VecchiaMLE.vecchia_generate_hess_tri_structure!(nnzh::Int, n::Int, colptr_diff::CuVector{Int}, 
                                                         hrows::CuVector{Int}, hcols::CuVector{Int})
    # reset hrows, hcols
    fill!(hrows, one(Int))
    fill!(hcols, one(Int))

    # launch the kernel
    backend = KA.get_backend(hrows)
    kernel = VecchiaMLE.vecchia_generate_hess_tri_structure_kernel!(backend)

    f(x) = (x * (x+1)) ÷ 2

    # NOTE: Might be a race condition here. Solution is to store them in the first indices of each thread. 
    view(hrows, 2:n) .+= cumsum(f.(view(colptr_diff, 1:n-1)))
    view(hcols, 2:n) .+= cumsum(view(colptr_diff, 1:n-1))

    kernel(nnzh, n, colptr_diff, view(hrows, 1:n), view(hcols, 1:n), hrows, hcols, ndrange = n)
    KA.synchronize(backend)
    return nothing
end

function VecchiaMLE.generate_samples(MatCov::CuMatrix{Float64}, number_of_samples::Int, ::Val{:gpu})
    S = copy(MatCov)
    V = CUDA.randn(Float64, number_of_samples, size(S, 1))
    LinearAlgebra.LAPACK.potrf!('U', S)
    rmul!(V, UpperTriangular(S))
    return V
end

function VecchiaMLE.generate_samples(::CuMatrix{Float64}, ::Int, ::Val{arch}) where {arch}
    error("Unsupported architecture $arch for GPU matrix input.")
end

function VecchiaMLE.generate_samples(::CuMatrix{Float64}, ::Int, ::Val{:cpu})
    error("GPU matrix with arch=:cpu. Choose arch=:gpu or convert to Matrix on CPU.")
end

VecchiaMLE.generate_samples(MatCov::CuMatrix{Float64}, number_of_samples::Int; arch::Symbol=:gpu) = generate_samples(MatCov, number_of_samples, Val(arch))

end
