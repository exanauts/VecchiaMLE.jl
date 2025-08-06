struct VecchiaCacheJump
    samples_outerprod::Matrix{Float64}
    samples::Matrix{Float64}
    n::Int
    M::Int
    colptr::Vector{Int}
    nnzL::Int
    rowsL::Vector{Int}
    colsL::Vector{Int}
    diagL::Vector{Int}
    lambda::Float64
end

function create_vecchia_cache_jump(samples::AbstractMatrix, Sparsity, lambda)
    M = size(samples, 1)
    n = size(samples, 2)
    
    # SPARSITY PATTERN OF L IN COO FORMAT.
    rows, cols, colptr = Sparsity
    diag = colptr[1:end-1]
    
    # Swap the two around for a second
    nnz_L = length(rows)
    samples_outerprod = sum(samples[k, :] * samples[k, :]' for k in 1:M)

    return VecchiaCacheJump(samples_outerprod, samples, n, M, colptr, nnz_L, rows, cols, diag, lambda)
end

function obj_vecchia(w::AbstractVector, cache::VecchiaCacheJump)
    t1 = -cache.M * sum(w[(cache.nnzL+1):end])
    # This looks stupid, but its better than putting it on one line
    t2 = sum(
            sum(
                sum(
                    w[r] * cache.samples[k, cache.rowsL[r]]
                    for r in cache.colptr[j]:(cache.colptr[j+1] - 1)
                )^2
                for j in 1:cache.n
            )
            for k in 1:cache.M
        )

    t3 = sum(w[i]^2 for i in 1:cache.nnzL if !(i in cache.diagL))

    return t1 + 0.5 * t2 + 0.5 * cache.lambda * t3
end

function cons_vecchia(w::AbstractVector, cache::VecchiaCacheJump)
    return [exp(w[i]) - w[j] for (i, j) in zip((1:cache.n).+cache.nnzL, cache.diagL)]
end
