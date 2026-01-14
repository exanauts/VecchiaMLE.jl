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
rowsL, colptrL = sparsity_pattern(iVecchiaMLE)
model = VecchiaModel(rowsL, colptrL, input.samples; format=:csc, uplo=:L)

output = madnlp(model)
L = recover_factor(colptrL, rowsL, output.solution)
```
