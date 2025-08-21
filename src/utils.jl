export generate_samples, generate_MatCov, generate_xyGrid, generate_rectGrid

function covariance2D(ptset::AbstractVector, params::AbstractVector)::AbstractMatrix
    return Symmetric([BesselK.matern(x, y, params) for x in ptset, y in ptset])
end


"""
    Samples_Matrix = generate_samples(MatCov::AbstractMatrix, 
                                      number_of_samples::Int;
                                      arch::Symbol)

    Generate a number of samples according to the given Covariance Matrix MatCov.
    Note the samples are given as mean zero. 

## Input arguments

* `MatCov`: A Covariance Matrix, assumed positive definite;
* `n`: The length of one side of the Covariance matrix;
* `number_of_samples`: How many samples to return.

## Keyword Arguments
* `arch`: Either to perform the linear algebra on cpu or gpu. See ARCHITECTURES. Defaults to `:cpu`.

## Output arguments

* `Samples_Matrix` : A matrix of size (number_of_samples × n), where the rows are the i.i.d samples  
"""
function generate_samples(MatCov::AbstractMatrix, number_of_samples::Int, ::Val{:cpu})
    S = Matrix(copy(MatCov))
    V = randn(number_of_samples, size(S, 1))
    LinearAlgebra.LAPACK.potrf!('U', S)
    rmul!(V, UpperTriangular(S))
    return V
end

function generate_samples(MatCov::CUDA.CuMatrix{Float64}, number_of_samples::Int, ::Val{:gpu})
    S = copy(MatCov)
    V = CUDA.randn(Float64, number_of_samples, size(S, 1))
    F = cholesky!(S)
    rmul!(V, F.U)
    return V
end

function generate_samples(::AbstractMatrix, ::Int, ::Val{arch}) where {arch}
    error("Unsupported architecture $arch for CPU matrix input.")
end

function generate_samples(::CUDA.CuMatrix{Float64}, ::Int, ::Val{arch}) where {arch}
    error("Unsupported architecture $arch for GPU matrix input.")
end

function generate_samples(::CUDA.CuArray{Float64,2}, number_of_samples::Int, ::Val{:cpu})
    error("GPU matrix with arch=:cpu. Choose arch=:gpu or convert to Matrix on CPU.")
end

function generate_samples(::AbstractMatrix, ::Int, ::Val{:gpu})
    error("CPU matrix with arch=:gpu. Choose arch=:cpu or move data to CuArray.")
end

generate_samples(MatCov::AbstractMatrix, number_of_samples::Int; arch::Symbol=:cpu) = generate_samples(MatCov, number_of_samples, Val(arch))
generate_samples(MatCov::CUDA.CuMatrix{Float64}, number_of_samples::Int; arch::Symbol=:gpu) = generate_samples(MatCov, number_of_samples, Val(arch))

"""
    Covariance_Matrix = generate_MatCov(params::AbstractArray,
                                        ptset::AbstractVector)

    Generates a matern-like Covariance Matrix determined via the given paramters (params) and ptset. 
    See https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function. 
## Input arguments

* `params`: An array of length 3 that holds the parameters to the matern covariance kernel (σ, ρ, ν);
* `ptset`: A set of points in 2D space upon which we determine the indices of the Covariance matrix;

## Output arguments

* `Covariance_Matrix` : An n × n Symmetric Matern matrix, where n = length(ptset).
"""
function generate_MatCov(params::AbstractArray, ptset::AbstractVector)::Symmetric{Float64}
    return covariance2D(ptset, params)
end

"""
    Covariance_Matrix = generate_MatCov(n::Int, 
                                        params::AbstractArray)

    Generates a 2D matern-like Covariance Matrix determined via the given paramters (params). 
    A square grid with `n` elements will be generated and used in lieu of a given location set.
    See https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function. 
## Input arguments

* `n`: The dimension of the desired MatCov;
* `params`: An array of length 3 that holds the parameters to the matern covariance kernel (σ, ρ, ν);

## Output arguments

* `Covariance_Matrix` : An n × n Symmetric Matern matrix 
"""
generate_MatCov(n::Int, params::AbstractVector) = generate_MatCov(params::AbstractVector, generate_xyGrid(n::Int))

"""
    Covariance_Matrix = generate_MatCov(params::AbstractArray,
                                        ptset::AbstractMatrix)

    Generates a matern-like Covariance Matrix determined via the given paramters (params). 
    The matrix of locations will be parsed as a vector of vectors, where each location is a row vector. 
    See https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function. 
## Input arguments

* `params`: An array of length 3 that holds the parameters to the matern covariance kernel (σ, ρ, ν);
* `ptset` : A matrix filled with row vectors of locations. 

## Output arguments

* `Covariance_Matrix` : An n × n Symmetric Matern matrix, where n = size(ptset, 1).
"""
generate_MatCov(params::AbstractVector, ptset::AbstractMatrix) = generate_MatCov(params::AbstractVector, tovector(ptset))

"""
    xyGrid = generate_xyGrid(n::Int)

    A helper function to generate a point grid which partitions the positive unit square [0, 1] × [0, 1].
    NOTE: This function at larger n values (n > 100) causes ill conditioning of the genreated Covariance matrix!
    See generate_safe_xyGrid() for higher n values.

## Input arguments

* `n`: The length of the desired xyGrid;
## Output arguments

* `xyGrid` : The desired points in 2D space  
"""
function generate_xyGrid(n::Int)::AbstractVector
    @assert isqrt(n)^2 == n "n is not square!"
    m = isqrt(n) 
    grid1d = range(0.0, 1.0, length=m)
    return vec([[x[1], x[2]] for x in Iterators.product(grid1d, grid1d)])
end

"""
    xyGrid = generate_safe_xyGrid(n::Int)

    A helper function to generate a point grid which partitions the positive square [0, k] × [0, k], 
    where k = cld(n, 10). This should avoid ill conditioning of a Covariance Matrix generated by these points.
    
## Input arguments

* `n`: The length of the desired xyGrid;
## Output arguments

* `xyGrid` : The desired points in 2D space  
"""
function generate_safe_xyGrid(n::Int)::AbstractVector
    @assert isqrt(n)^2 == n "n is not square!"
    m = isqrt(n) 
    len = cld(m, 10)
    
    grid1d = range(0.0, len, length=m)
    return vec([[x[1], x[2]] for x in Iterators.product(grid1d, grid1d)])
end

"""
    rect = generate_rectGrid(dims::Tuple)

A helper function to generate a point grid which partitions the positive square [0, 1] × [0, 1] by dims[1] and dims[2]. 

## Input arguments

* `dims`: A tuple of dimensions `(nx, ny)`;
## Output arguments

* `rectGrid` : The desired points in 2D space as a vector of 2-element arrays
"""
function generate_rectGrid(dims::Tuple{Int, Int})::AbstractVector
    nx, ny = dims
    @assert_cond nx > 0 nx "be positive"
    @assert_cond ny > 0 ny "be positive"

    grid_x = range(0.0, 1.0, length=nx)
    grid_y = range(0.0, 1.0, length=ny)
    return vec([[x, y] for x in grid_x, y in grid_y])
end

"""
    print_diagnostics(d::Diagnostics)

    Pretty prints the diagnostics of the VecchiaMLE Algorithm. 
    
## Input arguments
* `d`: The Diagnostics returned by the VecchiaMLE_Run function, assuming skip_check wasn't set to true.

"""
function print_diagnostics(d::Diagnostics)
    println("========== Diagnostics ==========")
    println(rpad("Model Creation Time:",  25), d.create_model_time)
    println(rpad("LinAlg Solve Time:",    25), d.linalg_solve_time)
    println(rpad("Solve Model Time:",     25), d.solve_model_time)
    println(rpad("Objective Value:",      25), d.objective_value)
    println(rpad("Normed Constraint:",    25), d.normed_constraint_value)
    println(rpad("Normed Gradient:",      25), d.normed_grad_value)
    println(rpad("Optimization Iter:",    25), d.iterations)
    println("=================================")
end
