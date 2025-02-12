# VecchiaMLE

## Overview

This project computes an approximation of the precision matrix's Cholesky factor for a covariance matrix using a Maximum Likelihood Estimation (MLE) formulation. The computed Cholesky factor, `L`, is generated via the Vecchia Approximation, which has the advantage of being **sparse** and **approximately banded**, making it efficient for large-scale problems. 

The only parameter which needs serious input is the samples. The samples matrix assumes that the samples are given as **row** vectors, not columns! 
Note that if the samples matrix is empty or is not given as an input, the analysis cannot be ran. 

> **Note:** All samples (generated or provided) are assumed to have a zero-mean distribution.

The code is written to be flexible enough to be run on either CPU or GPU capable systems, however only CUDA capable GPU's are supported at the moment.

## Installation and Dependencies

Before you start, make sure you have the following:

- **Julia**: Version 1.x or higher.
- **CUDA**: Required for GPU mode (if you plan on using it).
- **MadNLP**: Ensure that this nonlinear optimization solver is installed.
- **Probably more, Idk atm**

Other dependencies will be needed, but at this point I don't know which ones are seriously needed. TO BE REWRITTEN. 

## Configuration
The struct `VecchiaMLEInput` needs to be properly filled out in order for the analysis to be run. The components of said structure is as follows:

```
n::Integer                           # Square root of the size of the problem (e.g., sqrt dimension of the covariance matrix). 
k::Integer                           # Number of conditioning points per point for the Vecchia Approximation.
samples::Matrix{Float64}             # Matrix of samples (each row is a sample).
Number_of_Samples::Integer           # Number of samples to generate (if samples_given=false).
mode::Integer                        # Operation mode. Expects an int [1: CPU, 2: GPU].
MadNLP_Print_level::Integer          # Print level of MadNLP. Expects an int with the corresponding flag [1: TRACE, 2: DEBUG, 3: INFO, 4: WARN, 5: ERROR].
```

## Usage
Once `VecchiaMLEInput` has been filled appropriately, pass it to VecchiaMLE_Run() for the analysis to start. Note that some arguments have default values, such as mode (CPU), and MadNLP_Print_level (5). After the analysis has been completed, the function outputs diagnostics - that would be difficult other wise to acquire - and the resulting Lm factor in sparse, LowerTriangular format.

## Contribution
Although the bulk of the project has been written, there are sure to be problems that arise from errors in logic. As such, please feel free to open an issue;
we would appreciate it!


[![Build Status](https://github.com/CalebDerrickson/VecchiaMLE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/CalebDerrickson/VecchiaMLE.jl/actions/workflows/CI.yml?query=branch%3Amaster)
