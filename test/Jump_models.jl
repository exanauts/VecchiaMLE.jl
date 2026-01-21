function obj_vecchia(w::AbstractVector, samples, lambda, cache::VecchiaCache)
    t1 = -cache.M * sum(w[(cache.nnzL+1):end])
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

    t3 = sum(w[i]^2 for i in 1:cache.nnzL)

    return t1 + 0.5 * t2 + 0.5 * lambda * t3
end

function cons_vecchia(w::AbstractVector, cache::VecchiaCache)
    return [exp(w[i]) - w[j] for (i, j) in zip((1:cache.n).+cache.nnzL, cache.diagL)]
end
