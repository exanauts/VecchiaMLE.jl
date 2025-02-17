using VecchiaMLE

# Things for model
ns = 10:10:200
ns = 10:10:10
k = 10
Number_of_Samples = 100
params = [5.0, 0.2, 2.25, 0.25]
timings = zeros(2, length(ns))

for (i, n) in enumerate(ns)
    # Generate samples
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)
    input = VecchiaMLE.VecchiaMLEInput(0, k, samples, Number_of_Samples, 5, 1)
    input.samples = samples
    input.n = n

    # CPU
    input.mode = 1
    d_cpu, L_cpu = VecchiaMLE_Run(input)

    # GPU
    input.mode = 2
    d_gpu, L_gpu = VecchiaMLE_Run(input)
end
