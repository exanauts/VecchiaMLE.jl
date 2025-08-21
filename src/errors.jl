# TODO: Move errors to own file. Call it errors.jl
"""
    KL_Divergence = KLDivergence(TCov::Symmetric{Float64},
                                 AL::AbstractMatrix)

    Computes the KL Divergence of the True Covariance matrix, TCov, and
    The APPROXIMATE INVERSE CHOLESKY FACTOR, AL. Assumed mean zero.
    Note: This is extremely slow!

## Input arguments

* `TCov`: The True Covariance Matrix;
* `AL`: The cholesky factor to the approximate precision matrix. I.e., The ouptut of VecchiaMLE.  
## Output arguments

* `KL_Divergence`: The result of the KL Divergence function.
"""
function KLDivergence(TCov::Symmetric{Float64}, AL::AbstractMatrix)
    terms = zeros(4)
    terms[1] = tr(AL'*TCov*AL)
    terms[2] = -size(TCov, 1)
    terms[3] = -2*sum(log.(diag(AL)))
    terms[4] = -logdet(TCov)
    return 0.5*sum(terms)
end

"""
KL_Divergence = KLDivergence(TChol::T,
AChol::T) where {T <: AbstractMatrix}

Computes the KL Divergence of the cholesky of the True Covariance matrix, TChol, and
The APPROXIMATE INVERSE CHOLESKY FACTOR (The output of VecchiaMLE), AChol. Assumed mean zero.

## Input arguments

* `TChol`: The cholesky of the True Covariance Matrix;
* `AChol`: The cholesky factor to the approximate precision matrix. I.e., The ouptut of VecchiaMLE.  

## Output arguments

* `KL_Divergence`: The result of the KL Divergence function.
"""
function KLDivergence(TChol::T, AChol::T) where {T <: AbstractMatrix}
    terms = zeros(3)
    M = zeros(size(TChol, 1))
    for i in 1:size(TChol, 1)
        mul!(M, AChol, view(TChol, :, i))
        terms[1] += dot(M, M)
    end
    terms[2] = -size(TChol, 1)
    terms[3] = 2*sum(log.(1.0./(diag(AChol) .* diag(TChol))))
    return 0.5*sum(terms)
end

"""
    Error = uni_error(TCov::AbstractMatrix,
                      L::AbstractMatrix)
    Generates the "Univariate KL Divergence" from the Given True Covariane matrix 
    and Approximate Covariance matrix. The output is the result of the following 
    optimization problem: sup(a^T X_A || a^⊺ X_T). The solution is 
                f(μ) = ln(μ) + (2μ)^{-2} - 0.5,
    where μ is the largest or smallest eigenvalue of the matrix Σ_A^{-1}Σ_T, whichever
    maximizes the function f(μ). Note: Σ_A , Σ_T are the respective approximate and true
    covariance matrices, and X_A, X_T are samples from the respective Distributions (mean zero).    

## Input arguments

* `TCov`: The True Covariance Matrix: the one we are approximating;
* `L`: The cholesky factor of the approximate precision matrix.
## Output arguments

* `f(μ)` : the result of the function f(μ) detailed above.   
"""
function uni_error(TCov::AbstractMatrix, L::AbstractMatrix)    
    mu_val = clamp.([eigmin(L*L'*TCov), eigmax(L*L'*TCov)], 1e-12, Inf)
    return maximum([0.5*(log(mu) + 1.0 / mu - 1.0) for mu in mu_val])
end
