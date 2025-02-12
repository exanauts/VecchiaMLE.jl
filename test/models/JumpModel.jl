


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
    return t1 + 0.5 * t2
end

function grad_vecchia(g::AbstractVector, w::AbstractVector, cache::VecchiaCacheJump)
    g[1:n^2] = Vec_from_LowerTriangular(cache.samples_outerprod * L, cache.n)
    println("Geez")
    display(g)
    return g
end

function hess_vecchia(H::AbstractMatrix, w::AbstractVector, cache::VecchiaCacheJump)
    H[1:cache.n^2, 1:cache.n^2] = cache.samples_outerprod
    return H
end

function cons_vecchia(w::AbstractVector, cache::VecchiaCacheJump)
    #return exp.(w[(1:cache.n).+cache.nnzL]) .- w[cache.colptr[1:end-1]]
    return [exp(w[i]) - w[j] for  (i, j) in zip((1:cache.n).+cache.nnzL, cache.colptr[1:end-1])]
end

function jac_vecchia(grad_c::AbstractVector, w::AbstractVector, cache::VecchiaCacheJump)
    for (idx, i) in enumerate(cache.colptr[1:end-1])
        grad_c[i] -= y[idx] - L[idx, idx]
    end 
    return grad_c
end

function cons_hess_vecchia(Hess_c::AbstractMatrix, w::AbstractVector, cache::VecchiaCacheJump)

    # Diagonal term
    for (idx, i) in enumerate(cache.colptr[1:end-1])
        Hess_c[i, i] += lambda / (L[idx, idx])^2
    end

    # Lambda, L term
    lm_loc = Int(cache.n * (cache.n + 1) / 2) + cache.n
    for i in 1:cache.n
        for idx in eachindex(cache.colptr[1:end-1])
            Hess_c[lm_loc + i, idx] -= 1.0 / (L[idx, idx])
            Hess_c[idx, lm_loc + i] -= 1.0 / (L[idx, idx])
        end
    end
    
    # Y term
    y_loc = lm_loc - cache.n
    Hess_c[1:cache.n .+ y_loc, 1:cache.n] .+= 1.0
    Hess_c[1:cache.n, 1:cache.n .+ y_loc] .+= 1.0

    return Hess_c
end