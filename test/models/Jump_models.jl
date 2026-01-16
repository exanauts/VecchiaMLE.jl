
#=
struct VecchiaCacheJump
    samples::Matrix{Float64}
    n::Int
    M::Int
    colptrL::Vector{Int}
    nnzL::Int
    rowsL::Vector{Int}
    diagL::Vector{Int}
    lambda::Float64
    uplo::Symbol
end

function create_vecchia_cache_jump(samples::AbstractMatrix, Sparsity, lambda, uplo::Symbol)
    M = size(samples, 1)
    n = size(samples, 2)
    
    # SPARSITY PATTERN OF L IN COO FORMAT.
    rows, colptr = Sparsity
    if uplo == :L
        diag = colptr[1:end-1]
    else
        diag = colptr[2:end]
        diag .-= 1
    end
    # Swap the two around for a second
    nnz_L = length(rows)

    return VecchiaCacheJump(samples, n, M, colptr, nnz_L, rows, diag, lambda, uplo)
end
=#

function obj_vecchia(w::AbstractVector, samples, lambda, cache::VecchiaCache)
    t1 = -cache.M * sum(w[(cache.nnzL+1):end])
    # This looks stupid, but its better than putting it on one line
    t2 = sum(
            sum(
                sum(
                    w[r] * samples[k, cache.rowsL[r]]
                    for r in cache.colptrL[j]:(cache.colptrL[j+1] - 1)
                )^2
                for j in 1:cache.n
            )
            for k in 1:cache.M
        )

    t3 = sum(w[i]^2 for i in 1:cache.nnzL if !(i in cache.diagL))

    return t1 + 0.5 * t2 + 0.5 * lambda * t3
end

function cons_vecchia(w::AbstractVector, cache::VecchiaCache)
    return [exp(w[i]) - w[j] for (i, j) in zip((1:cache.n).+cache.nnzL, cache.diagL)]
end
