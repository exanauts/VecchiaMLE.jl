# How To Run An Analysis

This example shows how to run VecchiaMLE for a given VecchiaMLEInput. There are 
some necessary parameters the user needs to input, which are the following:

```@example HowToRun
using VecchiaMLE
using SparseArrays # To show matrix at the end 

# Things for model
n = 400
k = 10
Number_of_Samples = 100
```

These are the three major parameters: n (dimension), k (conditioning for Vecchia), 
and the Number_of_Samples. they should be self-explanatory, but the 
documentation for the VecchiaMLEInput should clear things up. Next, we should 
generate the samples. At the moment, the code has only been verified to run for
generated samples by the program, but there should be no difficulty in inputting
your own samples. Just make sure the length of each sample should be a perfect square.
This is a strange restriction, and will be removed when we allow user input locations.
Nevertheless, let's move on. 

The sample generation requires a covariance matrix. Here, we will generate a matern like
covairance matrix which VecchiaMLE has a predefined function for. Although not exposed
directly to the user, we allow the generation of samples in the following manner:

```@example HowToRun
params = [5.0, 0.2, 2.25, 0.25]
# This analysis will be done in 2D.
ptGrid = VecchiaMLE.generate_safe_xyGrid(n)
MatCov = VecchiaMLE.generate_MatCov(params, ptGrid)
samples = VecchiaMLE.generate_Samples(MatCov, Number_of_Samples)
```

Next, we create and fill in the VecchiaMLEInput struct. This is done below.

```@example HowToRun
input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1; ptGrid=ptGrid)
```

Here, we set (though not necessary) the optimizer (MadNLP) print level to 5 (ERROR), 
which keep it silent. Furthermore, we set the compute mode to 1 (cpu). The other option 
would be to set it to 2 (gpu), but since not all machines have a gpu connected, we opt
for the sure-fire approach.

All that's left is to run the analysis. This is done in one line:

```@example HowToRun
d, o = VecchiaMLE_Run(input)
```

We can see the structure of the output of the program has an approximately banded structure. 
This is due to the generation of the sparsity pattern depends on the euclidean distance between
any given point and previously considered points in the grid.

```@example HowToRun
sparse(o)
```  

The function `VecchiaMLE_Run` also returns some diagnostics that would be difficult otherwise to
retrieve, and the result of the analysis. This is the cholesky factor to the approximate precision
matrix. You can check the KL Divergence of this approximation, with a function inside VecchiaMLE, 
though this is not recommended for larger dimensions (n >= 30). This obviously requires the true
covariance matrix, which should be generated here if not before. 

```@example HowToRun
println("Error: ", VecchiaMLE.KLDivergence(MatCov, o))
```
