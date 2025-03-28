using NPZ
using VecchiaMLE
using DelimitedFiles
using CUDA

function main(; cpu::Bool=true, gpu::Bool=true)
    # The samples are given as a 3d array. Reshaping to 2d
    samples = NPZ.npzread("../../data/caleb_samples.npy")
    samples = reshape(samples, size(samples, 1), :)

    # Fist 100 samples.
    samples_cpu = Matrix{Float64}(samples[1:100, :])

    Number_of_Samples = size(samples_cpu, 1)
    dim = size(samples_cpu, 2)
    ns = collect(10000:10000:dim)
    k = 10

    if cpu
    time_memory_io_file =  open("time_memory_cpu.txt","a")
    write(time_memory_io_file, "time, memory\n")

    for (i, n) in enumerate(ns)
        nn = Int(floor(sqrt(n)))
        input = VecchiaMLE.VecchiaMLEInput(nn, min(nn, k), samples_cpu[:, 1:nn^2], Number_of_Samples, 5, 1)
        diagsnostics = @timed VecchiaMLE_Run(input)
        
        write(time_memory_io_file, "$(diagsnostics[2]), $(diagsnostics[3])\n")
        println("$i, cpu, $(diagsnostics[2]) s, \t $(round(Float64(i)/length(ns)*100.0, digits=4))%")
        if diagsnostics[2] > 300
            break 
        end
    end
    close(time_memory_io_file)
    end

    # CPU tests done, onto GPU tests
    samples_gpu = CuMatrix{Float64}(samples_cpu)

    if gpu
    time_memory_io_file =  open("time_memory_gpu.txt","a")
    write(time_memory_io_file, "time, memory\n")

    for (i, n) in enumerate(ns)
        nn = floor(sqrt(n))
        input = VecchiaMLE.VecchiaMLEInput(nn, min(nn, k), samples_gpu[:, 1:nn^2], Number_of_Samples, 5, 2)
        diagsnostics = @timed VecchiaMLE_Run(input)

        write(time_memory_io_file, "$(diagsnostics[2]), $(diagsnostics[3])\n")
        println("$i, gpu, $(diagsnostics[2]) s, \t $(round(Float64(i)/length(ns)*100.0, digits=4))%")
        
        if diagsnostics[2] > 300
            break 
        end
    end
    close(time_memory_io_file)
    end
end
