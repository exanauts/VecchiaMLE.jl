using VecchiaMLE, CUDA, DelimitedFiles

# Parameters
# ns = 10:10:200
ns = Vector{Int}(10:5:100)
ns .= ns.^2
k = 10
number_of_samples = 100
params = [5.0, 0.2, 2.25, 0.25]
timings_linalg = zeros(2, ns |> length)
timings_model = zeros(2, ns |> length)
timings_solve = zeros(2, ns |> length)

for (i, n) in enumerate(ns)
    # Generate samples
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples, :cpu)
    input_cpu = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples)
    input_gpu = VecchiaMLE.VecchiaMLEInput(n, k, CuMatrix(samples), number_of_samples; arch=:gpu)

    # cpu
    diagnostics_cpu, L_cpu = VecchiaMLE_Run(input_cpu)
    timings_model[1, i] = diagnostics_cpu.create_model_time
    timings_solve[1, i] = diagnostics_cpu.solve_model_time
    timings_linalg[1, i] = diagnostics_cpu.linalg_solve_time

    # gpu
    if CUDA.has_cuda()
        diagnostics_gpu, L_gpu = VecchiaMLE_Run(input_gpu)
        timings_model[2, i] = diagnostics_gpu.create_model_time
        timings_solve[2, i] = diagnostics_gpu.solve_model_time
        timings_linalg[2, i] = diagnostics_gpu.linalg_solve_time
    end

    println("$(round(( (i + 1) / length(ns) * 100.0);digits=4))")
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
