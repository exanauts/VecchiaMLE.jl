export generate_Samples, generate_MatCov, generate_xyGrid

"""
    Covariance_Matrix = covariance2D(ptGrid::AbstractVector, 
                                     params::AbstractVector)

    Generate a Matern-Like Covariance Matrix for the parameters and locations given.
    Note: This should not be called by the user. This is the back end for the function
    generate_MatCov()!

## Input arguments

* `ptGrid`: A set of points in 2D space upon which we determine the indices of the Covariance matrix;
* `params`: An array of length 3 (or 4) that holds the parameters to the matern covariance kernel (σ, ρ, ν).

## Output arguments

* `Covariance_Matrix` : An n × n Matern matrix, where n is the length of the ptGrid 
"""
function covariance2D(ptGrid::AbstractVector, params::AbstractVector)::AbstractMatrix
    return Symmetric([BesselK.matern(x, y, params) for x in ptGrid, y in ptGrid])
end


"""
    Samples_Matrix = generate_Samples(MatCov::AbstractMatrix, 
                                      n::Int,
                                      Number_of_Samples::Int;
                                      mode::VecchiaMLE.ComputeMode )

    Generate a number of samples according to the given Covariance Matrix MatCov.
    Note the samples are given as mean zero. 
    If a CUDA compatible device is detected, the samples are generated on the gpu
    and transferred back to the cpu. 

## Input arguments

* `MatCov`: A Covariance Matrix, presumably positive definite;
* `n`: The length of one side of the Covariance matrix;
* `Number_of_Samples`: How many samples to return.

## Keyword Arguments
* `mode`: Either generate samples on gpu or cpu

## Output arguments

* `Samples_Matrix` : A matrix of size (Number_of_Samples × n), where the rows are the i.i.d samples  
"""
function generate_Samples(MatCov::AbstractMatrix, n::Int, Number_of_Samples::Int; mode::ComputeMode=cpu)::AbstractMatrix
    if mode == gpu
        V = CUDA.randn(Float64, Number_of_Samples, n^2)
        S = CuArray{Float64}(MatCov)
    elseif mode == cpu 
        V = randn(Number_of_Samples, n^2)
        S = Matrix{Float64}(MatCov)
    else
        error("Unsupported compute mode!")
    end

    LinearAlgebra.LAPACK.potrf!('U', S)
    LinearAlgebra.rmul!(V, UpperTriangular(S))
    return V
end

"""
    Covariance_Matrix = generate_MatCov(n::Int,
                                        params::AbstractArray,
                                        ptGrid::AbstractVector)

    Generates a Matern Covariance Matrix determined via the given paramters (params) and ptGrid.

## Input arguments

* `n`: The length of the ptGrid;
* `params`: An array of length 3 (or 4) that holds the parameters to the matern covariance kernel (σ, ρ, ν);
* `ptGrid`: A set of points in 2D space upon which we determine the indices of the Covariance matrix;

## Output arguments

* `Covariance_Matrix` : An n × n Symmetric Matern matrix 
"""
function generate_MatCov(n::Int, params::AbstractArray, ptGrid::AbstractVector)::Symmetric{Float64}
    return covariance2D(ptGrid, params)
end

generate_MatCov(n::Int, params::AbstractVector) = generate_MatCov(n::Int, params::AbstractVector, generate_xyGrid(n::Integer))

"""
    xyGrid = generate_xyGrid(n::Integer)

    A helper function to generate a point grid which partitions the positive unit square [0, 1] × [0, 1].
    NOTE: This function at larger n values (n > 100) causes ill conditioning of the genreated Covariance matrix!
    See generate_safe_xyGrid() for higher n values.

## Input arguments

* `n`: The length of the desired xyGrid;
## Output arguments

* `xyGrid` : The desired points in 2D space  
"""
function generate_xyGrid(n::Int)::AbstractVector
    grid1d = range(0.0, 1.0, length=n)
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
    len = cld(n, 10)
    grid1d = range(0.0, len, length=n)
    return vec([[x[1], x[2]] for x in Iterators.product(grid1d, grid1d)])
end

"""
    model = get_vecchia_model(iVecchiaMLE::VecchiaMLEInput)

    creates and returns a vecchia model based on the VecchiaMLEInput and point grid. 
## Input arguments

* `iVecchiaMLE`: The filled out VecchiaMLEInput struct
## Output arguments

* `model`: The Vecchia model based on the VecchiaMLEInput  
"""
function get_vecchia_model(iVecchiaMLE::VecchiaMLEInput)::VecchiaModel


    if iVecchiaMLE.mode == gpu
       return VecchiaModelGPU(iVecchiaMLE.samples, iVecchiaMLE)
    else 
       return VecchiaModelCPU(iVecchiaMLE.samples, iVecchiaMLE)
    end
end

"""
    log_level = _printlevel(pLevel::Union{Int, PrintLevel})

    A helper function to convert an Int to a MadNLP LogLevel.
    The mapping is [1: 'TRACE', 2: 'DEBUG', 3: 'INFO', 4: 'WARN', 5: 'ERROR'].
    Any other Int given is converted to MadNLP.Fatal. 
## Input arguments

* `pLevel`: The given log level;
## Output arguments

* `log_level` : The coded MadNLP.LogLevel.   
"""
function _printlevel(pLevel::PL) where {PL <: Union{Int, PrintLevel}}
    get(PRINT_LEVEL_TO_MADNLP, pLevel, MadNLP.ERROR)
end

"""
    Dict to map VecchiaMLE Loglevel to MadNLP LogLevel. Also for Ints. 
"""
const PRINT_LEVEL_TO_MADNLP = Dict(
    VTRACE => MadNLP.TRACE,
    VDEBUG => MadNLP.DEBUG,
    VINFO  => MadNLP.INFO,
    VWARN  => MadNLP.WARN,
    VERROR => MadNLP.ERROR,
    VFATAL => MadNLP.ERROR,
    1      => MadNLP.TRACE,
    2      => MadNLP.DEBUG,
    3      => MadNLP.INFO,
    4      => MadNLP.WARN,
    5      => MadNLP.ERROR
)


"""
    cpu_mode = Int_to_Mode(n::Int)

    A helper function to convert an Int to a ComputeMode.
    The mapping is [1: 'cpu', 2: 'gpu'].
    Any other Int given is converted cpu.

## Input arguments

* `n`: The given cpu mode as an Int;
## Output arguments

* `cpu_mode` : The coded ComputeMode.   
"""
function _computemode(n::CM)::ComputeMode where {CM <: Union{ComputeMode, Int}}
    if isa(n, ComputeMode)
        return n
    end

    if n == 1
        return cpu
    elseif n == 2
        return gpu
    end
    return cpu
end

"""
    Error = Uni_Error(TCov::AbstractMatrix,
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
function Uni_Error(TCov::AbstractMatrix, L::AbstractMatrix)    
    mu_val = clamp.([eigmin(L*L'*TCov), eigmax(L*L'*TCov)], 1e-9, Inf)
    return maximum([0.5*(log(mu) + 1.0 / mu - 1.0) for mu in mu_val])
end


"""

    indx_perm, dist_set = IndexReorder(Cond_set::AbstractVector,
                                       data::AbstractVector,
                                       mean_mu::AbstractVector,
                                       method::String = "random", 
                                       reverse_ordering::Bool = true)

    Front End function to determine the reordering of the given indices (data).
    At the moment, only random, and maxmin are implemented.    

    I BELIEVE the intention is to permute the coordinates of the samples with it, e.g.,
    samples = samples[:, indx_perm]. Note for small conditioning sets per point (extremely sparse L),
    this is not worth while!.

## Input arguments

* `Cond_set`: A set of points from which to neglect from the index permutation;
* `data`: The data to determine said permutation;
* `mean_mu`: The theoretical mean of the data, not mean(data)!
* `method`: Either permuting based on maxmin, or random.
* `reverse_ordering`: permutation is generated either in reverse or not.   
## Output arguments

* `indx_perm`: The index set which permutes the given data.
* `dist_set`: The array of maximum distances to iterating conditioning sets.    
"""
function IndexReorder(Cond_set::AbstractVector, data::AbstractVector, mean_mu::AbstractVector, method::String="random", reverse_ordering::Bool=true)::AbstractVector
    
    if method == "random"
        return IndexRandom(Cond_set, data, reverse_ordering)
    elseif method == "maxmin"
        return IndexMaxMin(Cond_set, data, mean_mu, reverse_ordering)
    else 
        # Do Random
        return IndexRandom(Cond_set, data, reverse_ordering)
    end
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
function IndexMaxMin(Cond_set::AbstractVector, data::AbstractVector, mean_mu::AbstractVector, reverse_ordering::Bool=true)::AbstractVector
    n = size(data, 1)
    
    # distance array
    dist_arr = zeros(Float64, n)
    # index array
    index_arr = zeros(Int, n)
    
    idx = reverse_ordering ? n : 1

    # Finding last index
    if isempty(Cond_set)
        # Set the last entry arbitrarily, try to place in center
        # This means set it to data sample closest to the mean. 
        last_idx = 0
        min_norm = Inf64
        for i in 1:n
            data_norm = norm(mean_mu - data[i, :])
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
            data_pt = data[i, :]
            
            mindistance = dist_from_set(data_pt, Cond_set, data)
        
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
    chosen_inds = union(Cond_set, index_arr[reverse_ordering ? end : 1])
    remaining_inds = setdiff(1:n, chosen_inds)

    while(!isempty(remaining_inds))
        dist_max = 0.0
        idx_max = 0
        for j in remaining_inds
            data_pt = data[j, :]
            distance = dist_from_set(data_pt, chosen_inds, data)
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
end #function

"""
Helper function. See IndexReorder().
"""
function dist_from_set(data_pt, set_idx, data)

    mindistance = Inf64
    # Find distance from conditioning set
    for j in set_idx
        dist = norm(data_pt - data[j, :])
        if mindistance > dist
            mindistance = dist
        end
    end
    return mindistance
end


#
#    Just gives a random reordering pattern. 
#    Conditioning set will just be assumed empty.
#    Also returning nothing for the distance array (useless?)
#
"""
See IndexReorder().
"""
function IndexRandom(Cond_set::AbstractVector, data::AbstractVector, reverse_ordering::Bool)
    
    n = size(data, 1)
    
    reorder_idx = [i for i in 1:n]
    return Random.randperm(n), []
end
#    KLDivergence for the true covariance, TCov, and the 
#    cholesky factor for the approximate precision matrix, AL.
#    This is most applicable to the analysis at hand.
#    ASSUMED ZERO MEAN.

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




"""
    rows, cols, colptr = SparsityPattern(data::AbstractVector,
                                         k::Int,
                                         format::String = "")

    Front end to generate the sparsity pattern of the approximate 
    precision's cholesky factor, L, in CSC format. The pattern is 
    determined by nearest neighbors of the previous points in data.

## Input arguments
* `data`: The point grid that was used to either generate the covariance matrix, or any custom ptGrid.
* `k`: The number of nearest neighbors for each point (Including the point itself)
* `format`: Either blank or "CSC". This can be ignored.

## Output arguments
* `rows`: A vector of row indices of the sparsity pattern for L, in CSC format.
* `cols`: A vector of column indices of the sparsity pattern for L, in CSC format.
* `colptr`: A vector of incides which determine where new columns start. 

"""
function SparsityPattern(data, k::Int, format="")
    if format == ""
        return SparsityPattern_CSC(data, k)
    elseif format == "CSC"
        return SparsityPattern_CSC(data, k)
    else
        println("Sparsity Pattern: Bad format. Gave", format)
        return nothing
    end
end

"""
See SparsityPattern().
"""
function SparsityPattern_Block(data, k::Int)
    n = size(data, 1)
    Sparsity = Vector{Vector{Int}}(undef, n)  

    Sparsity[1] = [1]  
    kdtree = KNN.KDTree(reshape(Vector(data[1, 1]), :, 1))
    for i in 2:n
        neighbors, _ = AdaptiveKDTrees.KNN.knn(kdtree, data[i, 1], min(i-1, k))
        Sparsity[i] = [i; neighbors] 
        add_point!(kdtree, data[i, 1])
    end

    return Sparsity
end

"""
See SparsityPattern().
"""
function SparsityPattern_CSC(data, k::Int)
    n = size(data, 1)
    rows = zeros(Int, Int(0.5 * k * (2*n - k + 1)))
    cols = copy(rows)  
    
    Sparsity_dict = Dict(i => Vector{Int}(undef, 0) for i in 1:n) 
    Sparsity_dict[1] = [1]
    
    kdtree = KNN.KDTree(reshape(Vector(data[1, 1]), :, 1))
    
    buffer = Vector{Int}(undef, k)  
    
    for i in 2:n
        k_nn = min(i-1, k-1)
        buffer, _ = AdaptiveKDTrees.KNN.knn(kdtree, data[i, 1], k_nn)
        
        # Push to i-th point itself, since it is a neighbor.
        push!(Sparsity_dict[i], i)
    

        for j in 1:k_nn
            push!(Sparsity_dict[buffer[j]], i)
        end 
    
        # Add point to kdtree. 
        add_point!(kdtree, data[i, 1])
    end
    
    #println("Sparsity_dict:\n", Sparsity_dict)

    idx = 1
    colptr = zeros(Int, n+1)
    colptr[1] = 1
    for i in 1:n
        spar_i = Sparsity_dict[i]
        len = length(spar_i)
        cols[idx:idx+len-1] .= i
        rows[idx:idx+len-1] .= spar_i
        idx += len
        colptr[i+1] = idx
    end
    #println("idx:", idx)
    rows = rows[1:idx-1]
    cols = cols[1:idx-1]

    return rows, cols, colptr
end

"""
    sanitize_input!(iVecchiaMLE::VecchiaMLEInput, ptGrid::Union{AbstractVector, Nothing})

A helper function to catch any inconsistencies in the input given by the user. 
Note that if ptGrid is set as nothing, then the ptGrid is set as an equispaced mesh of grid points in [0, 1] x [0, 1]. 

The current checks are:\n
    * Ensuring n > 0.
    * Ensuring k <= n^2 (Makes sense considering the ptGrid and SparsityPattern sizes).
    * Ensuring the sample matrix, if the user gives one, is nonempty and is the same size as n^2.
    * Ensuring the pLevel in 1:5. See MadNLP_Print_Level().
    * Ensuring the mode in 1:2. See Int_to_Mode().

## Input arguments
* `iVecchiaMLE`: The filled-out VecchiaMLEInput struct. See VecchiaMLEInput struct for more details. 
"""
function sanitize_input!(iVecchiaMLE::VecchiaMLEInput) 
    @assert iVecchiaMLE.n > 0 "The dimension n must be strictly positive!"
    @assert iVecchiaMLE.k <= iVecchiaMLE.n^2 "The number of conditioning neighbors must be less than n^2 !"
    @assert size(iVecchiaMLE.samples, 1) > 0 "Samples must be nonempty!"
    @assert size(iVecchiaMLE.samples, 2) == iVecchiaMLE.n^2 "samples must be of size Number_of_Samples x n^2!"
    @assert size(iVecchiaMLE.samples, 1) == iVecchiaMLE.Number_of_Samples "samples must be of size Number_of_Samples x n^2!"
    
    if typeof(iVecchiaMLE.samples) <: Matrix && iVecchiaMLE.mode == gpu
        iVecchiaMLE.samples = CuMatrix{Float64}(iVecchiaMLE.samples)
    end

    @assert length(iVecchiaMLE.ptGrid) == iVecchiaMLE.n^2  "The ptGrid given does not have n^2 elements!"
    for (i, pt) in enumerate(iVecchiaMLE.ptGrid)
        @assert length(pt) == 2 "Position $(i) in ptGrid is not 2 dimensional!"
    end
end
