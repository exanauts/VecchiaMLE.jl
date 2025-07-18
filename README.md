# VecchiaMLE

[![docs-dev][docs-dev-img]][docs-dev-url] [![ci][ci-img]][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://exanauts.github.io/VecchiaMLE.jl/dev
[ci-img]: https://github.com/exanauts/VecchiaMLE.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/exanauts/VecchiaMLE.jl/actions/workflows/CI.yml

## Overview

This project computes an approximation of the precision matrix's Cholesky factor for a covariance matrix using a Maximum Likelihood Estimation (MLE) formulation. The computed Cholesky factor, `L`, is generated via the Vecchia Approximation, which has the advantage of being **sparse** and **approximately banded**, making it efficient for large-scale problems. 

The only parameter which needs serious input is the samples. The samples matrix assumes that the samples are given as **row** vectors, not columns! 
Note that if the samples matrix is empty or is not given as an input, the analysis cannot be ran. 

> **Note:** All samples (generated or provided) are assumed to have a zero-mean distribution.

The code is written to be flexible enough to be run on either cpu or gpu capable systems, however only CUDA capable gpu's are supported at the moment.

## Installation and Dependencies

Before you start, make sure you have the following:

- **Julia**: Version 1.x or higher.
- **CUDA**: Required for gpu mode (if you plan on using it).
- **MadNLP, MadNLPGPU**: Ensure that this nonlinear optimization solver is installed.
- **NearestNeighbors**: For use in generating the sparsity pattern for the Cholesky factor
- **NLPModels**: Construction of the Optimization problem.

## Configuration
The struct `VecchiaMLEInput` needs to be properly filled out in order for the analysis to be run. The components of said structure is as follows:

```
n::Integer                           # The size of the problem (e.g., dimension of the covariance matrix). 
k::Integer                           # Number of conditioning points per point for the Vecchia Approximation.
samples::Matrix{Float64}             # Matrix of samples (each row is a sample).
Number_of_Samples::Integer           # Number of samples to generate (if samples_given=false).
mode::Integer                        # Operation mode. Expects an int [1: cpu, 2: gpu].
MadNLP_Print_level::Integer          # Print level of MadNLP. Expects an int with the corresponding flag [1: TRACE, 2: DEBUG, 3: INFO, 4: WARN, 5: ERROR].
```

## Usage
Once `VecchiaMLEInput` has been filled appropriately, pass it to VecchiaMLE_Run() for the analysis to start. Note that some arguments have default values, such as mode (cpu), and MadNLP_Print_level (5). After the analysis has been completed, the function outputs diagnostics - that would be difficult other wise to acquire - and the resulting Lm factor in sparse, LowerTriangular format. 

> If the user desires to input their own location grid, then it must be passed as a keyword argument to VecchiaMLE_Run(). That is, `VecchiaMLE_Run(iVecchiaMLE::VecchiaMLEInput; ptGrid::AbstractVector)`. 

## Getting Samples from a Covariance Matrix
I will describe here how to properly use this package. Some functions used are not exported since there is no need for the user to realistically use them. The only major work to do is to generate the samples (if this isn't done by another means). In production, I generated the samples via first creating a Covariance Matrix via the martern covariance kernel, then feeding it into a Multivariate normal distribution to create the samples. The code to do this, using functions defined in VecchiaMLE, is as follows:

```
n = 10                                         # or any positive integer
Number_of_Samples = 100                        # or however many you want
params = [5.0, 0.2, 2.25, 0.25]                # Follow the procedure for matern in BesselK.jl
ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
MatCov = VecchiaMLE.generate_MatCov(params, ptGrid) # size n x n
samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples)
```

You can easily skip the Covariance generation if you already have one. To give insight as to why the covariance matrix is of that size, the creation of the covariance matrix requires a set of points in space to generate the matrix entries. This is done by generating a 2D grid, on the postive unit square. That is, we use the following function:

```
function covariance2D(ptGrid::AbstractVector, params::AbstractVector)::AbstractMatrix
    return Symmetric([BesselK.matern(x, y, params) for x in ptGrid, y in ptGrid])
end
```
The matern function (provided by BesselK, credit to Chris Geoga) generates the entries of the covariance matrix via the given prarmeters, and returns the symmetric form.

After the samples have been generated, they can simply be stored in the VecchiaMLEInput struct you intend to input into the program. Note that the resulting matrix, the cholesky factor of the precision matrix, is given as a LowerTriangular matrix, which does not clearly show its sparsity. However, the LowerTriangular format can be more easily leveraged by common operations (for example, solving systems and matrix inversion). 

## Getting the error

We can get the error for the approximation (assuming you have the true covariance matrix), via the KL-Divergence formula. This, along with its univariate cousin, can be queryed respectively by the following VecchiaMLE functions:

```
uni_error = VecchiaMLE.Uni_Error(True_Covariance, Approximate_Cholesky_Factor)
kl_error = VecchiaMLE.KLDivergence(True_Covariance, Approximate_Cholesky_Factor)
```
Note the KL-Divergence error is computationally heavy, thus takes a long time for large `n` values! Also, we assue mean-zero distributions. 

## Contribution
Although the bulk of the project has been written, there are sure to be problems that arise from errors in logic. As such, please feel free to open an issue;
we would appreciate it!
