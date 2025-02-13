export generate_Samples, generate_MatCov, generate_xyGrid

function covariance2D(xyGrid::AbstractVector, params::AbstractVector)::AbstractMatrix
    return Symmetric([matern(x, y, params) for x in xyGrid, y in xyGrid])
end

#function generate_Samples(MatCov::AbstractMatrix, n::Integer, Number_of_Samples::Integer)::Matrix{Float64}
#    mean_mu = zeros(n^2)
#    foo = MvNormal(mean_mu, MatCov)
#
#    # rand gives each samples as a column vector
#    return reshape(rand(foo, Number_of_Samples), (Number_of_Samples, n^2))
#end

# Added new generate_Samples to use GPU when there is one detected. Faster? Only problem is communitcation b/w GPU and CPU

function generate_Samples(MatCov::AbstractMatrix, n::Integer, Number_of_Samples::Integer)::Matrix{Float64}
    if CUDA.has_cuda()
        V = CUDA.rand(Number_of_Samples, n^2)
        S = CuArray{Float64}(MatCov)
    else 
        V = randn(Number_of_Samples, n^2)
        S = Matrix{Float64}(MatCov)
    end
    LinearAlgebra.LAPACK.potrf!('L', S)
    LinearAlgebra.rmul!(V, LowerTriangular(S))
    return Matrix{Float64}(V)
end

function generate_MatCov(n::Integer, params::AbstractArray, ptGrid::AbstractVector)::Symmetric{Float64}
    return covariance2D(ptGrid, params)
end

generate_MatCov(n::Integer, params::AbstractVector) = generate_MatCov(n::Integer, params::AbstractVector, generate_xyGrid(n::Integer))

function generate_xyGrid(n::Integer)::AbstractVector
    grid1d = range(0.0, 1.0, length=n)
    return vec([[x[1], x[2]] for x in Iterators.product(grid1d, grid1d)])
end

function MadNLP_Print_Level(pLevel::Int)::MadNLP.LogLevels
    if pLevel == 1
        return MadNLP.TRACE
    elseif pLevel == 2
        return MadNLP.DEBUG
    elseif pLevel == 3
        return MadNLP.INFO
    elseif pLevel == 4
        return MadNLP.WARN
    elseif pLevel == 5
        return MadNLP.ERROR
    else 
        return MadNLP.Fatal
    end
end

function Int_to_Mode(n::Int)::COMPUTE_MODE
    if n == 1
        return CPU
    elseif n == 2
        return GPU
    end
end

function Uni_Error(TCov::AbstractMatrix, L::AbstractMatrix)    
    mu_val = clamp.([eigmin(L*L'*TCov), eigmax(L*L'*TCov)], 1e-9, Inf)
    return maximum([0.5*(log(mu) + 1.0 / mu - 1.0) for mu in mu_val])
end

#
#    Determines reordering of given observation locations.
#    Included are Random indexing and maxmin
#
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
function IndexRandom(Cond_set::AbstractVector, data::AbstractVector, reverse_ordering::Bool)
    
    n = size(data, 1)
    
    reorder_idx = [i for i in 1:n]
    return Random.randperm(n), []
end


#
#    KLDivergence for the true covariance, TCov, and the 
#    cholesky factor for the approximate precision matrix, AL.
#    This is most applicable to the analysis at hand.
#    ASSUMED ZERO MEAN.
#
function KLDivergence(TCov::Symmetric{Float64}, AL)
    terms = zeros(4)
    terms[1] = tr(AL'*TCov*AL)
    terms[2] = -size(TCov, 1)
    terms[3] = -2*sum(log.(diag(AL)))
    terms[4] = -logdet(TCov)
    return 0.5*sum(terms)
end

#
#    Abstraction to get Sparsity pattern of Vecchia Approximation.
#    Will usually only be called for the SparsityPattern_CSC function
#    Options are Block or CSC.
#


function SparsityPattern(data, k::Int, format="")
    if format == ""
        return SparsityPattern_Block(data, k)
    elseif format == "CSC"
        return SparsityPattern_CSC(data, k)
    else
        println("Sparsity Pattern: Bad format. Gave", format)
        return nothing
    end
end

#
#    Will return the sparsity pattern in
#    row, [column] format. 
#    Bad name, but don't know what else to call it. 
#
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

#
#    Returns the sparsity pattern for the given data in COO, CSC format.
#
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
        spar_i = Sparsity_dict[i]  
        push!(spar_i, i)
    
        for j in 1:k_nn
            push!(Sparsity_dict[buffer[j]], i)
        end 
    
        add_point!(kdtree, data[i, 1])
    end
    
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

    return rows, cols, colptr
end

function sanitize_input!(iVecchiaMLE::VecchiaMLEInput)
    @assert iVecchiaMLE.n >= 0 "The dimension n must be strictly positive!"
    @assert iVecchiaMLE.k <= iVecchiaMLE.n^2 "The number of conditioning neighbors must be less than n^2 !"
    @assert size(iVecchiaMLE.samples, 1) > 0 "Samples must be nonempty!"
    @assert (iVecchiaMLE.MadNLP_print_level in 1:5) "MadNLP Print Level not in 1:5!"
    @assert size(iVecchiaMLE.samples, 2) == iVecchiaMLE.n^2 "samples must be of size Number_of_Samples x n^2!"
    @assert size(iVecchiaMLE.samples, 1) == iVecchiaMLE.Number_of_Samples "samples must be of size Number_of_Samples x n^2!"
    @assert iVecchiaMLE.mode in [1, 2] "Operation mode not valid! must be in [1, 2]." 
end
