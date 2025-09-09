# Running VecchiaMLE for Multiple Iterations

Let's say there is some analysis which requires the inverse cholesky of a Covariance matrix for each iteration. 
Since VecchiaMLE can be decomposed into two separate processes (to find the sparsity pattern of L, and to recover its entries),
and the set of locations does not change, we can skip the most computationally heavy part of VecchiaMLE. 

We generate a toy analysis to demonstrate this. 

```@example IterativeSolves
using VecchiaMLE
using SparseArrays # To show matrix at the end 

n = 100
k = 10
number_of_samples = 100
params = [5.0, 0.2, 2.25]
MatCov = VecchiaMLE.generate_MatCov(n, params)
samples = VecchiaMLE.generate_samples(MatCov, number_of_samples; arch=:cpu)

niter = 5

rowsL = ones(Int, Int(0.5 * k * (2*n - k + 1)))
colptrL = ones(Int, n+1)

for i in 1:niter
    if i == 1
        input = VecchiaMLEInput(k, samples)
    else
        input = VecchiaMLEInput(k, samples; rowsL=rowsL, colptrL=colptrL)
    end

    _, L = VecchiaMLE_Run(input)
    println("Iter $(i) Error: ", VecchiaMLE.KLDivergence(MatCov, L))

    # save sparsity pattern
    if i == 1
        global rowsL = L.rowval
        global colptrL = L.colptr
    end
end
```

Since the ptset didn't change over iterations, we can assume the sparsity pattern doesn't change. Therefore, 
we can save the sparsity pattern after the first iteration, and reuse it. This saves an enormous amount
of time, from personal anecdote. 