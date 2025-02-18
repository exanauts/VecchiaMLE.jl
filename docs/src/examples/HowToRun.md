# How To Run An Analysis

This example shows how to run VecchiaMLE for a given VecchiaMLEInput. There are 
some necessary parameters the user needs to input, which are the following:

```@example HowToRun
using VecchiaMLE

# Things for model
n = 10
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
MatCov = VecchiaMLE.generate_MatCov(n, params)
samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)
```

Next, we create and fill in the VecchiaMLEInput struct. This is done below.

```@example HowToRun
input = VecchiaMLE.VecchiaMLEInput(n, k, samples, Number_of_Samples, 5, 1)
```

Here, we set (though not necessary) the optimizer (MadNLP) print level to 5 (ERROR), 
which keep it silent. Furthermore, we set the compute mode to 1 (CPU). The other option 
would be to set it to 2 (GPU), but since not all machines have a GPU connected, we opt
for the sure-fire approach.

All that's left is to run the analysis. This is done in one line:

```@example HowToRun
d, o = VecchiaMLE_Run(input)
```

The function `VecchiaMLE_Run` returns some diagnostics that would be difficult otherwise to
retrieve, and the result of the analysis. This is the cholesky factor to the approximate precision
matrix. You can check the KL Divergence of this approximation, with a function inside VecchiaMLE, 
though this is not recommended for larger dimensions (n >= 30). This obviously requires the true
covariance matrix, which should be generated here if not before. 

```@example HowToRun
println("Error: ", VecchiaMLE.KLDivergence(MatCov, o))
```
