using Profile, PProf
using VecchiaMLE
using DelimitedFiles

Profile.init(n = 10^8, delay = 0.05)
# Only do cpu for now.
n = 100
k = 10
number_of_samples = 100
params = [5.0, 0.2, 2.25, 0.25]

@pprof begin
    MatCov = VecchiaMLE.generate_MatCov(n, params)
    samples = VecchiaMLE.generate_samples(MatCov, n, number_of_samples; arch=:cpu)
    input = VecchiaMLE.VecchiaMLEInput(n, k, samples, number_of_samples)

    global diagnostics, L = VecchiaMLE_Run(input)
    global error_KL = VecchiaMLE.KLDivergence(MatCov, L)
    global error_uni = VecchiaMLE.uni_error(MatCov, L)
      
end

# save stuff to file
labels_for_file = [
    "error_KL",
    "error_uni",
    "create_model_time",
    "linalg_solve_time", 
    "solve_model_time",
    "objective_value",
    "normed_constraint_value",
    "normed_grad_value",
    "iterations",
    "arch"
]


stuff_to_file = [error_KL, error_uni,
    diagnostics.create_model_time,
    diagnostics.linalg_solve_time,
    diagnostics.solve_model_time,
    diagnostics.objective_value,
    diagnostics.normed_constraint_value,
    diagnostics.normed_grad_value,
    diagnostics.iterations,
    diagnostics.arch,
]
stuff = hcat(labels_for_file, string.(stuff_to_file))

open("diagnostics.txt", "w") do io
    writedlm(io, stuff, ",")
end
