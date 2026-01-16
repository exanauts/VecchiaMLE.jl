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
