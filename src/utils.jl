export generate_samples, generate_MatCov, generate_xyGrid

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
    S = Matrix(MatCov)
    V = randn(number_of_samples, size(S, 1))
    LinearAlgebra.LAPACK.potrf!('U', S)
    rmul!(V, UpperTriangular(S))
    return V
end

function generate_samples(MatCov::CuArray{Float64}, number_of_samples::Int, ::Val{:gpu})
    S = CUDA.CuArray(MatCov) 
    V = CUDA.randn(Float64, number_of_samples, size(S, 1))
    F = cholesky!(S)
    rmul!(V, F.U)
    return V
end

function generate_samples(::AbstractMatrix, ::Int, ::Val{arch}) where {arch}
    error("Unsupported architecture $arch for CPU matrix input.")
end

function generate_samples(::CUDA.CuArray{T, 2} where T, ::Int, ::Val{arch}) where {arch}
    error("Unsupported architecture $arch for GPU matrix input.")
end

generate_samples(MatCov::AbstractMatrix, number_of_samples::Int; arch::Symbol=:cpu) = generate_samples(MatCov, number_of_samples, Val(arch))
generate_samples(MatCov::CuArray{<:Float64}, number_of_samples::Int; arch::Symbol=:gpu) = generate_samples(MatCov, number_of_samples, Val(arch))



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
    model = get_vecchia_model(iVecchiaMLE::VecchiaMLEInput)

    Creates and returns a vecchia model based on the VecchiaMLEInput and point grid. 
## Input arguments

* `iVecchiaMLE`: The filled out VecchiaMLEInput struct
## Output arguments

* `model`: The Vecchia model based on the VecchiaMLEInput  
"""
function get_vecchia_model(iVecchiaMLE::VecchiaMLEInput)::VecchiaModel
    return get_vecchia_model(iVecchiaMLE, Val(iVecchiaMLE.arch))
end

get_vecchia_model(iVecchiaMLE::VecchiaMLEInput, ::Val{:cpu}) = VecchiaModelCPU(iVecchiaMLE.samples, iVecchiaMLE)
get_vecchia_model(iVecchiaMLE::VecchiaMLEInput, ::Val{:gpu}) = VecchiaModelGPU(iVecchiaMLE.samples, iVecchiaMLE)


# TODO: Move error to own file. call it errors.jl
"""
    Error = uni_error(TCov::AbstractMatrix,
                      L::AbstractMatrix)
    Generates the "Univariate KL Divergence" from the Given True Covariane matrix 
    and Approximate Covariance matrix. The output is the result of the following 
    optimization problem: sup(a^T X_A || a^⊺ X_T). The solution is 
                f(μ) = ln(μ) + (2μ)^{-2} - 0.5,
    where μ is the largest or smallest eigenvalue of the matrix Σ_A^{-1}Σ_T, whichever
    maximizes the function f(μ). Note: Σ_A , Σ_T are the respective approximate and true
    covariance matrices, and X_A, X_T are samples from the respective Distributions (mean zero).    

## Input arguments

* `TCov`: The True Covariance Matrix: the one we are approximating;
* `L`: The cholesky factor of the approximate precision matrix.
## Output arguments

* `f(μ)` : the result of the function f(μ) detailed above.   
"""
function uni_error(TCov::AbstractMatrix, L::AbstractMatrix)    
    mu_val = clamp.([eigmin(L*L'*TCov), eigmax(L*L'*TCov)], 1e-12, Inf)
    return maximum([0.5*(log(mu) + 1.0 / mu - 1.0) for mu in mu_val])
end


# TODO: Move permutations to own file. call it permutations.jl
"""
    permutation, dists = IndexReorder(condset::AbstractVector,
                                       data::AbstractVector,
                                       mu::AbstractVector,
                                       method::String = "standard", 
                                       reverse_ordering::Bool = true)

    Front End function to determine the reordering of the given indices (data).
    At the moment, only random, and maxmin are implemented.    

    I BELIEVE the intention is to permute the coordinates of the samples with it, e.g.,
    samples = samples[:, permutation]. Note for small conditioning sets per point (extremely sparse L),
    this is not worth while!.

## Input arguments
* `condset`: A set of points from which to neglect from the index permutation;
* `data`: The data to determine said permutation;
* `mu`: The theoretical mean of the data, not mean(data)!
* `method`: Either permuting based on maxmin, or random.
* `reverse_ordering`: permutation is generated either in reverse or not.   

## Output arguments
* `permutation`: The index set which permutes the given data.
* `dists`: The array of maximum distances to iterating conditioning sets.    
"""
function IndexReorder(condset::AbstractVector, data::AbstractVector, mu::AbstractVector, method::String="standard", reverse_ordering::Bool=true)
    
    if method == "random"
        return IndexRandom(condset, data, reverse_ordering)
    elseif method == "maxmin"
        return IndexMaxMin(condset, data, mu, reverse_ordering)
    elseif method == "standard"
        len = length(data)
        return 1:len, []    
    end

    println("Invalid Reordering: ", method)
end

#=
    If the conditioning set is empty, arbitrarily pick a data entry.
    Data should be row vectors, collected vertically
    Condition set is assumed to be a part of the data,
    given as indices.

    The parameter reverse_ordering determines how the ordering is 
    given. This avoids having to reverse the array in the front end.
    For reverse_ordering = true, this will give back the reverse maxmin 
    ordering, as described in the OWHADI paper. 
    #TODO: NOT IMPLEMENTED FOR NON-EMPTY CONDITIONING SET. 
    #TODO: PROBABLY THE WORST STRETCH OF CODE IMAGINEABLE. REFACTOR?

    Returns the index reordering as well as the distances from chosen_inds. 
=#
"""
See IndexReorder().
"""
function IndexMaxMin(condset::AbstractVector, data::AbstractVector, mu::AbstractVector, reverse_ordering::Bool=true)
    n = size(data, 1)
    
    # distance array
    dist_arr = zeros(Float64, n)
    # index array
    index_arr = zeros(Int, n)
    
    idx = reverse_ordering ? n : 1

    # Finding last index
    if isempty(condset)
        # Set the last entry arbitrarily, try to place in center
        # This means set it to data sample closest to the mean. 
        last_idx = 0
        min_norm = Inf64
        for i in 1:n
            data_norm = norm(mu - data[i, :][1])
            if data_norm < min_norm
                min_norm = data_norm
                last_idx = i
            end
        end
        index_arr[idx] = last_idx
        dist_arr[idx] = min_norm
    else # Condition set is not empty
        ind_max = 1
        distance = Inf64
        for i in 1:n
            loc = data[i, :]
            
            mindistance = dist_from_set(loc, condset, data)
        
            # check for max
            if mindistance > distance
                distance = mindistance
                ind_max = i
            end
        
        end
        index_arr[idx] = ind_max
        dist_arr[idx] = distance
    end

    # apply succeeding idx
    idx += reverse_ordering ? -1 : 1
    
    # backfill index set
    # Create array with already chosen indices
    chosen_inds = union(condset, index_arr[reverse_ordering ? end : 1])
    remaining_inds = setdiff(1:n, chosen_inds)

    while(!isempty(remaining_inds))
        dist_max = 0.0
        idx_max = 0
        for j in remaining_inds
            loc = data[j, :]
            distance = dist_from_set(loc, chosen_inds, data)
            if dist_max < distance
                dist_max = distance
                idx_max = j
            end
        end 

        # update for next iteration
        index_arr[idx] = idx_max
        dist_arr[idx] = dist_max
        union!(chosen_inds, [idx_max])
        setdiff!(remaining_inds, [idx_max])
        
        idx += reverse_ordering ? -1 : 1
    end

    return index_arr, dist_arr
end

"""
Helper function. See IndexReorder().
"""
function dist_from_set(loc, setidxs, data)

    mindistance = Inf64
    # Find distance from conditioning set
    for j in setidxs
        dist = norm(loc - data[j, :])
        if mindistance > dist
            mindistance = dist
        end
    end
    return mindistance
end

"""
See IndexReorder().
"""
function IndexRandom(condset::AbstractVector, data::AbstractVector, reverse_ordering::Bool)
    n = size(data, 1)    
    return Random.randperm(n), []
end

# TODO: Move errors to own file. Call it errors.jl
"""
    KL_Divergence = KLDivergence(TCov::Symmetric{Float64},
                                 AL::AbstractMatrix)

    Computes the KL Divergence of the True Covariance matrix, TCov, and
    The APPROXIMATE INVERSE CHOLESKY FACTOR, AL. Assumed mean zero.
    Note: This is extremely slow!

## Input arguments

* `TCov`: The True Covariance Matrix;
* `AL`: The cholesky factor to the approximate precision matrix. I.e., The ouptut of VecchiaMLE.  
## Output arguments

* `KL_Divergence`: The result of the KL Divergence function.
"""
function KLDivergence(TCov::Symmetric{Float64}, AL::AbstractMatrix)
    terms = zeros(4)
    terms[1] = tr(AL'*TCov*AL)
    terms[2] = -size(TCov, 1)
    terms[3] = -2*sum(log.(diag(AL)))
    terms[4] = -logdet(TCov)
    return 0.5*sum(terms)
end

"""
KL_Divergence = KLDivergence(TChol::T,
AChol::T) where {T <: AbstractMatrix}

Computes the KL Divergence of the cholesky of the True Covariance matrix, TChol, and
The APPROXIMATE INVERSE CHOLESKY FACTOR (The output of VecchiaMLE), AChol. Assumed mean zero.

## Input arguments

* `TChol`: The cholesky of the True Covariance Matrix;
* `AChol`: The cholesky factor to the approximate precision matrix. I.e., The ouptut of VecchiaMLE.  

## Output arguments

* `KL_Divergence`: The result of the KL Divergence function.
"""
function KLDivergence(TChol::T, AChol::T) where {T <: AbstractMatrix}
    terms = zeros(3)
    M = zeros(size(TChol, 1))
    for i in 1:size(TChol, 1)
        mul!(M, AChol, view(TChol, :, i))
        terms[1] += dot(M, M)
    end
    terms[2] = -size(TChol, 1)
    terms[3] = 2*sum(log.(1.0./(diag(AChol) .* diag(TChol))))
    return 0.5*sum(terms)
end

# TODO: Move sparsity pattern stuff to own file. Call it sparsitypattern.jl
"""
    rows, cols, colptr = sparsitypattern(
        ::Val{<:Symbol}, 
        data::AbstractVector,
        k::Int,
        metric::Distances.Metric=Distances.Euclidean()
    )

    Generates the sparsity pattern of the approximate 
    precision's cholesky factor, L, in CSC format. The pattern is 
    determined by nearest neighbors of the previous points in data.

## Input arguments
* `data`: The point grid that was used to either generate the covariance matrix, or any custom ptset.
* `k`: The number of nearest neighbors for each point (Including the point itself)
* `method`: How the sparsity pattern is generated. See SPARSITY_GEN
* `metric`: The metric for determining nearest neighbors.

## Output arguments
* `rows`: A vector of row indices of the sparsity pattern for L, in CSC format.
* `cols`: A vector of column indices of the sparsity pattern for L, in CSC format.
* `colptr`: A vector of incides which determine where new columns start. 

"""
function sparsitypattern(::Val{:NN}, iVecchiaMLE::VecchiaMLEInput)
    return sparsitypattern_NN(iVecchiaMLE.ptset, iVecchiaMLE.k, iVecchiaMLE.metric)
end

function sparsitypattern(::Val{:HNSW}, iVecchiaMLE::VecchiaMLEInput)
    return sparsitypattern_HNSW(iVecchiaMLE.ptset, iVecchiaMLE.k, iVecchiaMLE.metric)
end

function sparsitypattern(::Val{:USERGIVEN}, iVecchiaMLE::VecchiaMLEInput)
    return iVecchiaMLE.rowsL, iVecchiaMLE.colsL, iVecchiaMLE.colptrL
end

function sparsitypattern(::Val{M}, iVecchiaMLE::VecchiaMLEInput) where {M}
    error("sparsitypattern: Bad method. Gave $M")
end

function sparsitypattern(::Val{:NN}, ptset::AbstractVector, k::Int, metric::Distances.Metric=Distances.Euclidean())
    return sparsitypattern_NN(ptset, k, metric)
end

function sparsitypattern(::Val{:HNSW}, ptset::AbstractVector, k::Int, metric::Distances.Metric=Distances.Euclidean())
    return sparsitypattern_HNSW(ptset, k, metric)
end

sparsitypattern(ptset::AbstractVector, k::Int) = sparsitypattern(Val(:NN), ptset, k)

"""
See sparsitypattern(). Uses NearestNeighbors library. In case of tie, opt for larger index. 
"""
function sparsitypattern_NN(data, k, metric::Distances.Metric=Distances.Euclidean())::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
    n = size(data, 1)
    sparsity = Matrix{Int}(undef, n, k)
    fill!(sparsity, -1)
    view(sparsity, :, 1) .= 1:n
    inds = Vector{Int}(undef, k)
    
    if k < 2 return nn_to_csc(sparsity) end
    
    # only weird part
    data = hcat(data...)

    for i in 2:n    
        knn = min(i-1, k-1)
    
        # Reform balltree.  
        balltree = NearestNeighbors.BallTree(data[:, 1:i-1], metric; leafsize = 1)

        view(inds, 1:knn) .= NearestNeighbors.knn(balltree, data[:, i], knn)[1]
        view(sparsity, i, 2:knn+1) .= view(inds, knn:-1:1)
    end
    return nn_to_csc(sparsity)
end

"""
See sparsitypattern(). In place for HNSW.jl
"""
function sparsitypattern_HNSW(data, k, metric::Distances.Metric=Distances.Euclidean())
    n = size(data, 1)
    sparsity = Matrix{Int}(undef, n, k)
    fill!(sparsity, -1)
    view(sparsity, :, 1) .= 1:n
    inds = Vector{Int}(undef, k)

    if k < 2 return nn_to_csc(sparsity) end

    #Intialize HNSW struct
    hnsw = HierarchicalNSW(data; metric=metric, efConstruction=100, M=16, ef=50)
    add_to_graph!(hnsw, 1)

    for i in 2:n    
        knn = min(i, k)

        add_to_graph!(hnsw, i)
        view(inds, 1:knn) .= knn_search(hnsw, i, knn)[1]        
        view(sparsity, i, 2:knn) .= view(inds, 2:knn)
        
    end

    return nn_to_csc(sparsity)
end

# TODO: Move to internals
"""
    Checks csc format. A user can call this, but not advised. 
"""
function is_csc_format(iVecchiaMLE::VecchiaMLEInput)::Bool
    rowsL = iVecchiaMLE.rowsL
    colsL = iVecchiaMLE.colsL
    colptrL = iVecchiaMLE.colptrL
    n = iVecchiaMLE.n

    # lengths
    length(rowsL) != length(colsL) && return false
    
    length(colptrL) != n + 1 && return false

    # colptr should be non-decreasing
    any(diff(colptrL) .< 0) && return false

    # indices in range
    !all(1 .<= rowsL .<= n) && return false
    !all(1 .<= colsL .<= n) && return false

    # check row ordering within each column
    for col in 1:n
        start_idx = colptrL[col]
        end_idx = colptrL[col+1]-1
        !issorted(rowsL[start_idx:end_idx]) && return false
    end

    return true
end

# TODO: Move validate_input to internals
"""
    validate_input(iVecchiaMLE::VecchiaMLEInput)

A helper function to catch any inconsistencies in the input given by the user.

## Input arguments
* `iVecchiaMLE`: The filled-out VecchiaMLEInput struct. See VecchiaMLEInput struct for more details. 
"""
function validate_input(iVecchiaMLE::VecchiaMLEInput) 

    @assert_cond iVecchiaMLE.n > 0 iVecchiaMLE.n "be strictly positive"
    @assert_cond_compare iVecchiaMLE.k <= iVecchiaMLE.n  
    @assert_cond size(iVecchiaMLE.samples, 1) > 0 iVecchiaMLE.samples "have at least one sample"
    @assert_eq size(iVecchiaMLE.samples, 2) iVecchiaMLE.n
    @assert_eq size(iVecchiaMLE.samples, 1) iVecchiaMLE.number_of_samples 
    @assert eltype(iVecchiaMLE.samples) <: AbstractFloat "samples must have eltype which is a subtype of AbstractFloat"

    if typeof(iVecchiaMLE.samples) <: Matrix && iVecchiaMLE.arch == :gpu
        @warn "architecture given is gpu, but samples are on cpu. Transferring samples to gpu."
        iVecchiaMLE.samples = CuMatrix{Float64}(iVecchiaMLE.samples)
    end

    @assert_eq length(iVecchiaMLE.ptset) iVecchiaMLE.n
    
    dimension = length(iVecchiaMLE.ptset[1])
    for pt in iVecchiaMLE.ptset
        @assert_eq length(pt) dimension 
    end

    if !isnothing(iVecchiaMLE.rowsL) && !isnothing(iVecchiaMLE.colsL) && !isnothing(iVecchiaMLE.colptrL)
        @assert is_csc_format(iVecchiaMLE) "rowsL and colsL are not in CSC format"
    end

    if !isnothing(iVecchiaMLE.lvar_diag)    
        @assert_eq length(iVecchiaMLE.lvar_diag) cache.n
    end
    
    if !isnothing(iVecchiaMLE.uvar_diag)
        @assert_eq length(iVecchiaMLE.uvar_diag) cache.n 
    end
    
    @assert_cond iVecchiaMLE.lambda >= 0 iVecchiaMLE.lambda "be positive"

    nvar = Int(0.5 * iVecchiaMLE.k * ( 2*iVecchiaMLE.n - iVecchiaMLE.k + 1))
    if !isnothing(iVecchiaMLE.x0)
        @assert_eq length(iVecchiaMLE.x0) nvar 
    end

    @assert_cond iVecchiaMLE.solver_tol > 0.0 iVecchiaMLE.solver_tol "be positive"

    @assert_in iVecchiaMLE.solver SUPPORTED_SOLVERS
    @assert_in iVecchiaMLE.arch ARCHITECTURES
    @assert_in iVecchiaMLE.plevel PRINT_LEVEL
    @assert_in iVecchiaMLE.sparsitygen SPARSITY_GEN

end

"""
    rows, cols, colptr = nn_to_csc(sparmat::Matrix{Float64})

    A helper function to generate the sparsity pattern of the vecchia approximation (inverse cholesky) based 
    on each point's nearest neighbors. If there are n points, each with k nearest neighbors, then the matrix
    sparmat should be of size n x k. 
    
    NOTE: The Nearest Neighbors algorithm should only consider points which appear before the given point. If
    you do standard nearest neighbors and hack off the indices greater than the row number, it will not work. 

    TODO: Can be parallelized. GPU kernel?
    
## Input arguments
* `sparmat`: The n x k matrix which for each row holds the indices of the nearest neighbors in the ptset.

## Output arguments
* `rows`: A vector of row indices of the sparsity pattern for L, in CSC format.
* `cols`: A vector of column indices of the sparsity pattern for L, in CSC format.
* `colptr`: A vector of incides which determine where new columns start. 

"""
function nn_to_csc(sparmat::Matrix{Int})::Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
    n, k = size(sparmat)
    
    # Preprocess the counts
    counts = zeros(Int, n)
    for i in 1:n
        knn = min(i, k)
        view(counts, view(sparmat, i, 1:knn)) .+= 1
    end

    # Preallocate spari
    spari = zeros(maximum(counts))
    rows = ones(Int, Int(0.5 * k * (2*n - k + 1)))
    cols = copy(rows)  
    idx = 0
    colptr = ones(Int, n+1)
    for i in 1:n
        knn = min(i, k)
        # Find all rows that contain i in it. TODO: Could be better?
        spari_idx = 1
        for j in i:n
            if i in view(sparmat, j, :)
                spari[spari_idx] = j
                spari_idx+=1
            end
        end

        len = counts[i]
        cols[(1:len).+idx] .= i
        rows[(1:len).+idx] .= view(spari, 1:len)
        idx += len

    end
    counts .= cumsum(counts)
    view(colptr, 2:n+1) .+= counts

    return rows, cols, colptr
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

# TODO: Move functions below to internals.jl
function vecchia_solver(::Val{s}, args...; kwargs...) where {s}
    error("The solver $s is not available.")
end

function vecchia_solver(::Val{:madnlp}, args...; kwargs...)
    madnlp(args...; kwargs...)
end


# MadNLP Conversions

function resolve_plevel(::Val{:madnlp}, plevel::Val{T}) where {T}
    error("Unsupported print level $(T) for solver :madnlp.")
end

function resolve_plevel(solver::Val{<:Symbol}, ::Val{T}) where {T}
    error("The solver $(solver) does not have defined print level $(T).")
end

resolve_plevel(::Val{:madnlp}, ::Val{:VTRACE}) = MadNLP.TRACE
resolve_plevel(::Val{:madnlp}, ::Val{:VDEBUG}) = MadNLP.DEBUG
resolve_plevel(::Val{:madnlp}, ::Val{:VINFO})  = MadNLP.INFO
resolve_plevel(::Val{:madnlp}, ::Val{:VWARN})  = MadNLP.WARN
resolve_plevel(::Val{:madnlp}, ::Val{:VERROR}) = MadNLP.ERROR
resolve_plevel(::Val{:madnlp}, ::Val{:VFATAL}) = MadNLP.ERROR

function convert_plevel(::Val{T}) where {T}
    error("Unsupported print level $T")
end

convert_plevel(::Val{:VTRACE}) = :VTRACE
convert_plevel(::Val{1}) = :VTRACE

convert_plevel(::Val{:VDEBUG}) = :VDEBUG
convert_plevel(::Val{2}) = :VDEBUG

convert_plevel(::Val{:VINFO}) = :VINFO
convert_plevel(::Val{3}) = :VINFO

convert_plevel(::Val{:VWARN}) = :VWARN
convert_plevel(::Val{4}) = :VWARN

convert_plevel(::Val{:VERROR}) = :VERROR
convert_plevel(::Val{5}) = :VERROR


function convert_computemode(::Val{arch}) where {arch}
    error("Unsupported architecture: $arch")
end

convert_computemode(::Val{:cpu}) = :cpu
convert_computemode(::Val{1})    = :cpu

convert_computemode(::Val{:gpu}) = :gpu
convert_computemode(::Val{2})    = :gpu


function tovector(A::AbstractMatrix)::AbstractVector
    return size(A, 1) > size(A, 2) ? [row for row in eachrow(A)] :
                                     [col for col in eachcol(A)]
end

function check_x0!(x0_::AbstractVector, iVecchiaMLE::VecchiaMLEInput, cache::VecchiaCache)
    if !isnothing(iVecchiaMLE.x0) 
        if mapreduce(x -> x > 0, &, view(iVecchiaMLE.x0, cache.diagL))       
            view(x0_, 1:cache.nnzL) .= iVecchiaMLE.x0
            view(x0_, (1:cache.n).+cache.nnzL) .= log.(view(iVecchiaMLE.x0, cache.diagL))
        else
            @warn "User given x0 is not feasible. Setting x0 such that the initial Vecchia approximation is the identity."
            view(x0_, cache.diagL) .= one(eltype(x0_))
        end
    end 
end

function check_lvar!(lvar::AbstractVector, iVecchiaMLE::VecchiaMLEInput, cache::VecchiaCache)
    if !isnothing(iVecchiaMLE.lvar_diag)
        view(lvar, cache.diagL) .= iVecchiaMLE.lvar_diag
        view(lvar, (1:cache.n).+cache.nnzL) .= log.(iVecchiaMLE.lvar_diag)
    else
        # Always ensure that the diagonal coefficient Lᵢᵢ of the Vecchia approximation are strictly positive
        view(lvar, cache.diagL) .= 1e-16
        view(lvar, (1:cache.n).+cache.nnzL) .= log(1e-16)
    end
end

function check_uvar!(uvar::AbstractVector, iVecchiaMLE::VecchiaMLEInput, cache::VecchiaCache)    
    if !isnothing(iVecchiaMLE.uvar_diag)
        view(uvar, cache.diagL) .= iVecchiaMLE.uvar_diag
        view(uvar, (1:cache.n).+cache.nnzL) .= log.(iVecchiaMLE.uvar_diag)
    end
end

resolve_ptset(n::Int, ::Nothing) = generate_safe_xyGrid(n)
resolve_ptset(::Int, ptset::AbstractMatrix) = tovector(ptset)
resolve_ptset(::Int, ptset::AbstractVector) = ptset

function resolve_sparistygen(V1::V, sparsitygen::Val{:SparsityGen}) where {V <: Union{AbstractVector, Nothing}} 
    if isnothing(V1) return sparsitygen end 
    return :USERGIVEN
end