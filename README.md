# VecchiaMLE

[![docs-dev][docs-dev-img]][docs-dev-url] [![ci][ci-img]][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://exanauts.github.io/VecchiaMLE.jl/dev
[ci-img]: https://github.com/exanauts/VecchiaMLE.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/exanauts/VecchiaMLE.jl/actions/workflows/CI.yml

## Overview

**VecchiaMLE.jl** is a Gaussian Process (GP) Machine Learning library written in Julia, which approximates the inverse cholesky factor of a GPs Covariance 
matrix via a nonparametric optimization process. The nonzero entries of the inverse cholesky, L, are determined via the Vecchia Approximation - that is, 
conditional depenencies are determined by local proximity to any given point. The values of L are recovered via optimizing the joint probability distribtuion 
function (mean zero) to best match the given samples. Other regularization terms are implemented for stability and biasing purposes.

**VecchiaMLE.jl** requires the user to provide at the least the samples matrix and the accuracy of the Vecchia Approximation (number of conditioning points, which manifests as the number of nonzeros per row of L). Other parameters are available, including (but no limited to) the option for GPU utilization, print level, and providing an initial point x0 to coax the optimizer to a solution.   

> **Note:** All samples (generated or provided) are assumed to have a zero-mean distribution.

The code is written to be flexible enough to be run on either cpu or gpu capable systems, however only CUDA capable gpu's are supported at the moment.

## Installation and Dependencies

**VecchiaMLE.jl** requires the following packages for startup. Others may be necessary, depending on your use case. 

- **CUDA**: Required for gpu mode (if you plan on using it).
- **MadNLP, MadNLPGPU**: We utilize the **MadNLP** optimizer as a default sovler. Other solvers available are **KNITRO** and **Ipopt**.
- **NearestNeighbors, HNSW**: Either package can be specified for computation of the sparsity pattern of the inverse Cholesky.
- **NLPModels**: Construction of the Optimization problem.

## Configuration
The struct `VecchiaMLEInput` is available for the user to specify their analysis. At the minimum, we require the following:

```
* k::Int                      # Number of conditioning points per point for the Vecchia Approximation.
* samples::Matrix{Float64}    # Matrix of samples (each row is a sample).
```
Other options may be passed as keyword arguments. Such options are:

```
* plevel::Symbol            # Print level for the optimizer. See PRINT_LEVEL. Defaults to `ERROR`.
* arch::Symbol              # Architecture for the analysis. See ARCHITECTURES. Defaults to `:cpu`.
* ptset::AbstractVector     # The locations of the analysis. May be passed as a matrix or vector of vectors.
* lvar_diag::AbstractVector # Lower bounds on the diagonal of the sparse Vecchia approximation.
* uvar_diag::AbstractVector # Upper bounds on the diagonal of the sparse Vecchia approximation.
* rowsL::AbstractVector     # The sparsity pattern rows of L if the user gives one. MUST BE IN CSC FORMAT! 
* colsL::AbstractVector     # The sparsity pattern cols of L if the user gives one. MUST BE IN CSC FORMAT!
* colptrL::AbstractVector   # The column pointer of L if the user gives one. MUST BE IN CSC FORMAT!
* solver::Symbol            # Optimization solver (:madnlp, :ipopt, :knitro). Defaults to `:madnlp`.
* solver_tol::Float64       # Tolerance for the optimization solver. Defaults to `1e-8`.
* skip_check::Bool          # Whether or not to skip the `validate_input` function.
* sparsityGeneration        # The method by which to generate a sparsity pattern. See SPARSITY_GEN.
* metric::Distances.metric  # The metric by which nearest neighbors are determined. Defaults to Euclidean.
* lambda::Real              # The regularization scalar for the ridge `0.5 * λ‖L - diag(L)‖²` in the objective. Defaults to 0.
* x0::AbstractVector        # The user may give an initial condition, but it is limiting if you do not have the sparsity pattern. 
```

## Usage
Once the user has initialized an instance of `VecchiaMLEInput`, they may pass it to `VecchiaMLE_Run()` to recover the inverse cholesky. The ouput of `VecchiaMLE_Run()` is a tuple, with vales internal diagnostics and the inverse cholesky in sparse COO format.  

## Getting Samples from a Covariance Matrix
I will describe here how to properly use this package. Some functions used are not exported since there is no need for the user to realistically use them. The only major work to do is to generate the samples (if this isn't done by another means). In production, I generated the samples via first creating a Covariance Matrix via the martern covariance kernel, then feeding it into a Multivariate normal distribution to create the samples. The code to do this, using functions defined in VecchiaMLE, is as follows:

```
n = 100                                                            # Dimension of Covariance matrix
number_of_samples = 100                                            # Analysis samples number
params = [5.0, 0.2, 2.25]                                          # Parameters for Matern matrix. See BesselK.jl. 
ptset = VecchiaMLE.generate_safe_xyGrid(n)                         # Generates 100 equiparitioned points on [0, 10] x [0, 10].
MatCov = VecchiaMLE.generate_MatCov(params, ptset)                 # size n x n. 
samples = VecchiaMLE.generate_samples(MatCov, number_of_samples)   # Generate samples. 
```

One may skip Covariance generation, as its use is only to generate samples. The matern function generates the entries of the covariance matrix via the given prarmeters, and returns the symmetric form.

## Getting the error

We can get the error for the approximation (assuming you have the true covariance matrix), via the KL-Divergence formula. An error derived on individual dimensions may also be queryed. These errors are found via the two functions:

```
uni_error = VecchiaMLE.uni_error(True_Covariance, Approximate_Cholesky_Factor)
kl_error = VecchiaMLE.KLDivergence(True_Covariance, Approximate_Cholesky_Factor)
```
Note the KL-Divergence error is computationally heavy, and is expensive even at `n = 900`. 

## Contribution
Although the bulk of the project has been written, there are sure to be problems that arise from errors in logic. As such, please feel free to open an issue;
we would appreciate it!
