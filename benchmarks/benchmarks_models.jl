using VecchiaMLE
using VecchiaMLE: cpu, gpu
using NLPModels
using CUDA

function benchmarks_models(; n::Int=10, k::Int=10, Number_of_Samples::Int=100, verbose::Bool=true)
  params = [5.0, 0.2, 2.25, 0.25] 
  xyGrid = VecchiaMLE.generate_xyGrid(n)
  MatCov = VecchiaMLE.generate_MatCov(n, params, xyGrid)

  samples_cpu = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=cpu)
  samples_gpu = VecchiaMLE.generate_Samples(MatCov, n, Number_of_Samples; mode=gpu)
  model_cpu = VecchiaMLE.VecchiaModelCPU(samples_cpu, k, xyGrid)  # warm up
  model_gpu = VecchiaMLE.VecchiaModelGPU(samples_gpu, k, xyGrid)  # warm up
  timer_model_cpu = @elapsed VecchiaMLE.VecchiaModelCPU(samples_cpu, k, xyGrid)
  timer_model_gpu = CUDA.@elapsed VecchiaMLE.VecchiaModelGPU(samples_gpu, k, xyGrid)
  ratio_timer_model = timer_model_cpu / timer_model_gpu
  verbose && println("-- Time to create the VecchiaModel --")
  verbose && println("cpu: $timer_model_cpu")
  verbose && println("gpu: $timer_model_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_model")
  verbose && println()

  x_cpu = get_x0(model_cpu)
  x_gpu = get_x0(model_gpu)
  obj(model_cpu, x_cpu)  # warm up
  obj(model_gpu, x_gpu)  # warm up 
  timer_obj_cpu = @elapsed obj(model_cpu, x_cpu)
  timer_obj_gpu = CUDA.@elapsed obj(model_gpu, x_gpu)
  ratio_timer_obj = timer_obj_cpu / timer_obj_gpu
  verbose && println("-- Time to evaluate the objective --")
  verbose && println("cpu: $timer_obj_cpu")
  verbose && println("gpu: $timer_obj_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_obj")
  verbose && println()

  g_cpu = similar(x_cpu)
  g_gpu = similar(x_gpu)
  grad!(model_cpu, x_cpu, g_cpu)  # warm up
  grad!(model_gpu, x_gpu, g_gpu)  # warm up
  timer_grad_cpu = @elapsed grad!(model_cpu, x_cpu, g_cpu)
  timer_grad_gpu = CUDA.@elapsed grad!(model_gpu, x_gpu, g_gpu)
  ratio_timer_grad = timer_grad_cpu / timer_grad_gpu
  verbose && println("-- Time to evaluate the gradient --")
  verbose && println("cpu: $timer_grad_cpu")
  verbose && println("gpu: $timer_grad_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_grad")
  verbose && println()

  c_cpu = Vector{eltype(x_cpu)}(undef, get_ncon(model_cpu))
  c_gpu = CuVector{eltype(x_gpu)}(undef, get_ncon(model_gpu))
  cons!(model_cpu, x_cpu, c_cpu)  # warm up
  cons!(model_gpu, x_gpu, c_gpu)  # warm up
  timer_cons_cpu = @elapsed cons!(model_cpu, x_cpu, c_cpu)
  timer_cons_gpu = CUDA.@elapsed cons!(model_gpu, x_gpu, c_gpu)
  ratio_timer_cons = timer_cons_cpu / timer_cons_gpu
  verbose && println("-- Time to evaluate the constraints --")
  verbose && println("cpu: $timer_cons_cpu")
  verbose && println("gpu: $timer_cons_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_cons")
  verbose && println()

  hrows_cpu = Vector{Int}(undef, model_cpu.meta.nnzh)
  hcols_cpu = Vector{Int}(undef, model_cpu.meta.nnzh)
  hrows_gpu = CuVector{Int}(undef, model_gpu.meta.nnzh)
  hcols_gpu = CuVector{Int}(undef, model_gpu.meta.nnzh)
  hess_structure!(model_cpu, hrows_cpu, hcols_cpu)  # warm up
  hess_structure!(model_gpu, hrows_gpu, hcols_gpu)  # warm up
  timer_hess_structure_cpu = @elapsed hess_structure!(model_cpu, hrows_cpu, hcols_cpu)
  timer_hess_structure_gpu = CUDA.@elapsed hess_structure!(model_gpu, hrows_gpu, hcols_gpu)
  ratio_timer_hess_structure = timer_hess_structure_cpu / timer_hess_structure_gpu
  verbose && println("-- Time to evaluate the structure of the Hessian --")
  verbose && println("cpu: $timer_hess_structure_cpu")
  verbose && println("gpu: $timer_hess_structure_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_hess_structure")
  verbose && println()

  hvals_cpu = Vector{eltype(x_cpu)}(undef, model_cpu.meta.nnzh)
  hvals_gpu = CuVector{eltype(x_gpu)}(undef, model_gpu.meta.nnzh)
  hess_coord!(model_cpu, x_cpu, hvals_cpu)  # warm up
  hess_coord!(model_gpu, x_gpu, hvals_gpu)  # warm up
  timer_hess_obj_cpu = @elapsed hess_coord!(model_cpu, x_cpu, hvals_cpu)
  timer_hess_obj_gpu = CUDA.@elapsed hess_coord!(model_gpu, x_gpu, hvals_gpu)
  ratio_timer_hess_obj = timer_hess_obj_cpu / timer_hess_obj_gpu
  verbose && println("-- Time to evaluate the Hessian of the objective --")
  verbose && println("cpu: $timer_hess_obj_cpu")
  verbose && println("gpu: $timer_hess_obj_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_hess_obj")
  verbose && println()

  y_cpu = get_y0(model_cpu)
  y_gpu = get_y0(model_gpu)
  hess_coord!(model_cpu, x_cpu, y_cpu, hvals_cpu)  # warm up
  hess_coord!(model_gpu, x_gpu, y_gpu, hvals_gpu)  # warm up
  timer_hess_lag_cpu = @elapsed hess_coord!(model_cpu, x_cpu, y_cpu, hvals_cpu)
  timer_hess_lag_gpu = CUDA.@elapsed hess_coord!(model_gpu, x_gpu, y_gpu, hvals_gpu)
  ratio_timer_hess_lag = timer_hess_lag_cpu / timer_hess_lag_gpu
  verbose && println("-- Time to evaluate the Hessian of the Lagrangian --")
  verbose && println("cpu: $timer_hess_lag_cpu")
  verbose && println("gpu: $timer_hess_lag_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_hess_lag")
  verbose && println()

  jrows_cpu = Vector{Int}(undef, model_cpu.meta.nnzj)
  jcols_cpu = Vector{Int}(undef, model_cpu.meta.nnzj)
  jrows_gpu = CuVector{Int}(undef, model_gpu.meta.nnzj)
  jcols_gpu = CuVector{Int}(undef, model_gpu.meta.nnzj)
  jac_structure!(model_cpu, jrows_cpu, jcols_cpu)  # warm up
  jac_structure!(model_gpu, jrows_gpu, jcols_gpu)  # warm up
  timer_jac_structure_cpu = @elapsed jac_structure!(model_cpu, jrows_cpu, jcols_cpu)
  timer_jac_structure_gpu = CUDA.@elapsed jac_structure!(model_gpu, jrows_gpu, jcols_gpu)
  ratio_timer_jac_structure = timer_jac_structure_cpu / timer_jac_structure_gpu
  verbose && println("-- Time to evaluate the structure of the Jacobian --")
  verbose && println("cpu: $timer_jac_structure_cpu")
  verbose && println("gpu: $timer_jac_structure_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_jac_structure")
  verbose && println()

  jvals_cpu = Vector{eltype(x_cpu)}(undef, model_cpu.meta.nnzj)
  jvals_gpu = CuVector{eltype(x_gpu)}(undef, model_gpu.meta.nnzj)
  jac_coord!(model_cpu, x_cpu, jvals_cpu)  # warm up
  jac_coord!(model_gpu, x_gpu, jvals_gpu)  # warm up
  timer_jac_cpu = @elapsed jac_coord!(model_cpu, x_cpu, jvals_cpu)
  timer_jac_gpu = CUDA.@elapsed jac_coord!(model_gpu, x_gpu, jvals_gpu)
  ratio_timer_jac = timer_jac_cpu / timer_jac_gpu
  verbose && println("-- Time to evaluate the Jacobian --")
  verbose && println("cpu: $timer_jac_cpu")
  verbose && println("gpu: $timer_jac_gpu")
  verbose && println("Ratio cpu / gpu: $ratio_timer_jac")
  verbose && println()

  timer_cpu = Dict(:model => timer_model_cpu,
                   :obj => timer_obj_cpu,
                   :grad! => timer_grad_cpu,
                   :cons! => timer_cons_cpu,
                   :jac_structure! => timer_jac_structure_cpu,
                   :hess_structure! => timer_hess_structure_cpu,
                   :jac_coord! => timer_jac_cpu,
                   :hess_coord! => timer_hess_obj_cpu,
                   :hess_lag_coord! => timer_hess_lag_cpu)

  timer_gpu = Dict(:model => timer_model_gpu,
                   :obj => timer_obj_gpu,
                   :grad! => timer_grad_gpu,
                   :cons! => timer_cons_gpu,
                   :jac_structure! => timer_jac_structure_gpu,
                   :hess_structure! => timer_hess_structure_gpu,
                   :jac_coord! => timer_jac_gpu,
                   :hess_coord! => timer_hess_obj_gpu,
                   :hess_lag_coord! => timer_hess_lag_gpu)

  timer_ratio = Dict(:model => ratio_timer_model,
                     :obj => ratio_timer_obj,
                     :grad! => ratio_timer_grad,
                     :cons! => ratio_timer_cons,
                     :jac_structure! => ratio_timer_jac_structure,
                     :hess_structure! => ratio_timer_hess_structure,
                     :jac_coord! => ratio_timer_jac,
                     :hess_coord! => ratio_timer_hess_obj,
                     :hess_lag_coord! => ratio_timer_hess_lag)

  return timer_cpu, timer_gpu, timer_ratio
end
