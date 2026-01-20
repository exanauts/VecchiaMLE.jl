# [NonparametricVecchia.jl documentation](@id Home)

## Overview

This package computes an approximate Cholesky factorization of a covariance matrix using a Maximum Likelihood Estimation (MLE) approach.
The Cholesky factor is computed via the Vecchia approximation, which is sparse and approximately banded, making it highly efficient for large-scale problems.

## Installation

```julia
julia> ]
pkg> add https://github.com/exanauts/NonparametricVecchia.jl.git
pkg> test NonparametricVecchia
```
