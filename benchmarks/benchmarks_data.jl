using NPZ
using VecchiaMLE
using DelimitedFiles
using CUDA

function main()
    ns = Vector{Int}(10:5:100)
    ns .= ns.^2
    k = 10
    number_of_samples = 100
    params = [5.0, 0.2, 2.25, 0.25]

    # Generate samples
    params = [5.0, 0.2, 2.25, 0.25]
    MatCov = VecchiaMLE.generate_MatCov(maximum(ns), params)
    samples = VecchiaMLE.generate_samples(MatCov, number_of_samples, :cpu)


    k = 10
    time_memory_io_file = open("time_memory_cpu.txt","a")
    write(time_memory_io_file, "time, memory\n")

    for (i, n) in enumerate(ns)
        input = VecchiaMLE.VecchiaMLEInput(n, min(n, k), samples[:, 1:n], number_of_samples)
        diagsnostics = @timed VecchiaMLE_Run(input)
        
        write(time_memory_io_file, "$(diagsnostics[2]), $(diagsnostics[3])\n")
        println("$i, cpu, $(diagsnostics[2]) s, \t $(round(Float64(i)/length(ns)*100.0, digits=4))%")
        if diagsnostics[2] > 25 
            break 
        end
    end
    close(time_memory_io_file)

    # cpu tests done, onto gpu tests
    samples = convert(Matrix{Float64}, samples)
    samples = CuMatrix{Float64}(samples)
    
    time_memory_io_file =  open("time_memory_gpu.txt","a")
    write(time_memory_io_file, "time, memory\n")
    
    # warm start for gpu
    input = VecchiaMLE.VecchiaMLEInput(25, 10, samples[:, 1:25], number_of_samples; arch=:gpu)
    diagsnostics = @timed VecchiaMLE_Run(input)

    for (i, n) in enumerate(ns)
        sample_slice = view(samples, :, 1:n)
        input = VecchiaMLE.VecchiaMLEInput(n, min(n, k), sample_slice, number_of_samples; arch=:gpu)
        diagsnostics = @timed VecchiaMLE_Run(input)

        write(time_memory_io_file, "$(diagsnostics[2]), $(diagsnostics[3])\n")
        println("$i, gpu, $(diagsnostics[2]) s, \t $(round(Float64(i)/length(ns)*100.0, digits=4))%")
        
        if diagsnostics[2] > 25 
            break 
        end
    end

    close(time_memory_io_file)
end
