# How To Reuse the Solution of VecchiaMLE 

This example shows how to run VecchiaMLE for a given VecchiaMLEInput from absolute scratch. That is, we start with
no samples, and no location set ptset. In this case, we define three parameters.

```@example HowToRun
using VecchiaMLE
using SparseArrays # To show matrix at the end 

# Things for model
n = 400                 # Dimension
k = 10                  # Conditioning
number_of_samples = 100 # Number of Samples to generate
```

These three parameters determine any given analysis. We rely on funcitons defined in 
VecchiaMLE, namely those to generate samples.   

The sample generation requires a covariance matrix. We generate a matern like
covariance matrix for our analysis. We allow the generation of samples in the following manner:

```@example HowToRun
params = [5.0, 0.2, 2.25, 0.25]
# This analysis will be done in 2D.
ptset = VecchiaMLE.generate_safe_xyGrid(n)
MatCov = VecchiaMLE.generate_MatCov(params, ptset)
samples = VecchiaMLE.generate_samples(MatCov, number_of_samples)
```

Above, we generate a set of locations (ptset), a covariance matrix (MatCov), and the samples.  
Next, we create and fill in the VecchiaMLEInput struct. This is done below.

```@example HowToRun
input = VecchiaMLE.VecchiaMLEInput(k, samples; ptset=ptset)
```

The constructor takes at the bare minimum the locations on which to condition any given point (k), 
as well as the samples matrix. Since a ptset was already generated, we go ahead and provide this. If 
a ptset is not given, we fall back on a default locations set defined by the function `generate_safe_xyGrid`. 

All that's left is to run the analysis. This is done in one line:

```@example HowToRun
d, L = VecchiaMLE_Run(input)
```

We can see the structure of the output of the program has an approximately banded structure. 
This is due to the underlying assumption of the Vecchia Approximation where locaitons are independent of 
eachother when conditioned on its k nearest neighbors. 

```@example HowToRun
sparse(o)
```  

The function `VecchiaMLE_Run` returns a tuple, respectively containing the diagnostics of the solver, and the inverse cholesky factor L.
A function to obtian the KL Divergence of this approximation, is provided via an internal function of VecchiaMLE.  

```@example HowToRun
println("Error: ", VecchiaMLE.KLDivergence(MatCov, L))
```

The diagnostics can be displayed via an internal function. 

```@example HowToRun
VecchiaMLE.print_diagnostics(d)
```
