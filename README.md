# VecchiaMLE

[![docs-dev][docs-dev-img]][docs-dev-url] [![ci][ci-img]][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://exanauts.github.io/VecchiaMLE.jl/dev
[ci-img]: https://github.com/exanauts/VecchiaMLE.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/exanauts/VecchiaMLE.jl/actions/workflows/CI.yml

## Overview

**VecchiaMLE.jl** is a Julia package that approximates the inverse cholesky factor of a Gaussian process covariance matrix via a nonparametric optimization process.
The nonzero entries of the inverse cholesky, `L`, are determined via the Vecchia Approximation.
The values of `L` are recovered via optimizing the joint probability distribution function (mean zero) to best match the given samples.

## Installation

This package is not registered but can be installed and tested through the Julia package manager:

```julia
julia> ]
pkg> add https://github.com/exanauts/VecchiaMLE.jl.git
pkg> test VecchiaMLE
```

## Configuration

The struct `VecchiaMLEInput` is available for the user to specify their analysis. At the minimum, we require the following:

```
* k::Int                      # Number of conditioning points per point for the Vecchia Approximation.
* samples::Matrix{Float64}    # Matrix of samples (each row is a sample).
```
Other options may be passed as keyword arguments. Such options are:

```
* ptset::AbstractVector     # The locations of the analysis. May be passed as a matrix or vector of vectors.
* rowsL::AbstractVector     # The sparsity pattern rows of L if the user gives one. MUST BE IN CSC FORMAT! 
* colptrL::AbstractVector   # The column pointer of L if the user gives one. MUST BE IN CSC FORMAT!
* sparsityGeneration        # The method by which to generate a sparsity pattern. See SPARSITY_GEN.
* metric::Distances.metric  # The metric by which nearest neighbors are determined. Defaults to Euclidean.
```

## Usage

Once the user has initialized an instance of `VecchiaMLEInput`, they may pass it to `sparsity_pattern` to recover the inverse cholesky.

## Getting samples from a covariance matrix

```
n = 100                                                            # Dimension of Covariance matrix
number_of_samples = 100                                            # Analysis samples number
params = [5.0, 0.2, 2.25]                                          # Parameters for Matern matrix. See BesselK.jl. 
ptset = VecchiaMLE.generate_safe_xyGrid(n)                         # Generates 100 equiparitioned points on [0, 10] x [0, 10].
MatCov = VecchiaMLE.generate_MatCov(params, ptset)                 # size n x n. 
samples = VecchiaMLE.generate_samples(MatCov, number_of_samples)   # Generate samples. 
```
