using NPZ
using VecchiaMLE
using DelimitedFiles
using CUDA

function main()
    # The samples are given as a 3d array. Reshaping to 2d
    samples = NPZ.npzread("caleb_samples.npy")
    samples = reshape(samples, size(samples, 1), :)
    # Fist 100 samples.
    samples = samples[1:100, 1:round(Int, sqrt(size(samples, 2)))^2]

    Number_of_Samples = size(samples, 1)
    ens = 2:size(samples, 2)
    ns = [x for x in ens if x == round(Int, (sqrt(x)))^2]

    k = 10
    time_memory_io_file =  open("time_memory_cpu.txt","a")
    write(time_memory_io_file, "time, memory\n")

    for (i, n) in enumerate(ns)
        input = VecchiaMLE.VecchiaMLEInput(n, min(n, k), samples[:, 1:n], Number_of_Samples, 5, 1)
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

    for (i, n) in enumerate(ns)
        sample_slice = view(samples, :, 1:n)
        input = VecchiaMLE.VecchiaMLEInput(n, min(n, k), sample_slice, Number_of_Samples, 5, 2)
        diagsnostics = @timed VecchiaMLE_Run(input)

        write(time_memory_io_file, "$(diagsnostics[2]), $(diagsnostics[3])\n")
        println("$i, gpu, $(diagsnostics[2]) s, \t $(round(Float64(i)/length(ns)*100.0, digits=4))%")
        
        if diagsnostics[2] > 25 
            break 
        end
    end

    close(time_memory_io_file)
end
