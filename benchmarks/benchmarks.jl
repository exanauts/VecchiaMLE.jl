using VecchiaMLE, CUDA, DelimitedFiles

# Parameters
# ns = 10:10:200
ns = 10:5:100
k = 10
Number_of_Samples = 100
params = [5.0, 0.2, 2.25, 0.25]
timings_linalg = zeros(2, ns |> length)
timings_model = zeros(2, ns |> length)
timings_solve = zeros(2, ns |> length)

for (i, n) in enumerate(ns)
    # Generate samples
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples)
    input = VecchiaMLE.VecchiaMLEInput(0, k, samples, Number_of_Samples, 5, 1)
    input.samples = samples
    input.n = n

    # CPU
    input.mode = 1
    diagnostics_cpu, L_cpu = VecchiaMLE_Run(input)
    timings_model[1, i] = diagnostics_cpu.create_model_time
    timings_solve[1, i] = diagnostics_cpu.solve_model_time
    timings_linalg[1, i] = diagnostics_cpu.LinAlg_solve_time

    # GPU
    if CUDA.has_cuda()
        input.mode = 2
        diagnostics_gpu, L_gpu = VecchiaMLE_Run(input)
        timings_model[2, i] = diagnostics_gpu.create_model_time
        timings_solve[2, i] = diagnostics_gpu.solve_model_time
        timings_linalg[2, i] = diagnostics_gpu.LinAlg_solve_time
    end
end

open("timings_linalg.txt", "w") do io
    writedlm(io, timings_linalg)
end

open("timings_model.txt", "w") do io
    writedlm(io, timings_model)
end

open("timings_solve.txt", "w") do io
    writedlm(io, timings_solve)
end
