


struct VecchiaCacheJump
    samples_outerprod::Matrix{Float64}
    samples::Matrix{Float64}
    n::Int
    M::Int
    colptr::Vector{Int}
    nnzL::Int
    rowsL::Vector{Int}
    colsL::Vector{Int}
end

function create_vecchia_cache_jump(samples::AbstractMatrix, Sparsity)
    M = size(samples, 1)
    n = size(samples, 2)
    
    # SPARSITY PATTERN OF L IN COO FORMAT.
    rows, cols, colptr = Sparsity 
    
    # Swap the two around for a second
    nnz_L = length(rows)
    samples_outerprod = sum(samples[k, :] * samples[k, :]' for k in 1:M)


    return VecchiaCacheJump(samples_outerprod, samples, n, M, colptr, nnz_L, rows, cols)
end

function obj_vecchia(w::AbstractVector, cache::VecchiaCacheJump, model, lambda::Real=0.0)
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
    expr = @expression(
        model,
        sum((w[i] - (i in view(cache.colptr, 1:cache.n) || i > cache.nnzL ? w[i] : 0.0))^2 for i in eachindex(w))
    )
    return t1 + 0.5 * t2 + (lambda * 0.5) * expr
end

function cons_vecchia(w::AbstractVector, cache::VecchiaCacheJump)
    #return exp.(w[(1:cache.n).+cache.nnzL]) .- w[cache.colptr[1:end-1]]
    return [exp(w[i]) - w[j] for  (i, j) in zip((1:cache.n).+cache.nnzL, cache.colptr[1:end-1])]
end
