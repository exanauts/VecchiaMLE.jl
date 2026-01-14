####################################################
#               resolve_rowsL              
####################################################

resolve_rowsL(rowsL::Nothing, n::Int, k::Int) = zeros(Int, Int(0.5 * k * ( 2*n - k + 1)))
resolve_rowsL(rowsL::AbstractVector, n::Int, k::Int) = rowsL

####################################################
#               resolve_colptrL             
####################################################

resolve_colptrL(colptrL::Nothing, n::Int) = zeros(Int, n+1)
resolve_colptrL(colptrL::AbstractVector, n::Int) = colptrL

####################################################
# tovector. 
# To convert ptset from Matrix to Vector of Vectors
####################################################

function tovector(A::AbstractMatrix)::AbstractVector
    return size(A, 1) > size(A, 2) ? [row for row in eachrow(A)] :
                                     [col for col in eachcol(A)]
end

####################################################
# resolve_ptset
# MD for resolving user given (or not) pset .   
####################################################
resolve_ptset(::Int, ptset) = error("Unsupported ptset type: $(typeof(ptset))")
resolve_ptset(n::Int, ::Nothing) = generate_safe_xyGrid(n)
resolve_ptset(::Int, ptset::AbstractMatrix) = tovector(ptset)
resolve_ptset(::Int, ptset::AbstractVector{<:AbstractVector}) = ptset

####################################################
# resolve_sparsity_generator
# MD for resolving user given (or not) 
# sparsity pattern.   
####################################################

# Cases where user gives a sparsity pattern (doesn't matter its contents, just as long as its of type AbstractVector{<:Unsigned} )
resolve_sparsity_generator(::AbstractVector{<:Unsigned}, ::Val{:NN}) = :USERGIVEN
resolve_sparsity_generator(::AbstractVector{<:Unsigned}, ::Val{:HNSW}) = :USERGIVEN
resolve_sparsity_generator(::AbstractVector{<:Unsigned}, ::Val{:USERGIVEN}) = :USERGIVEN
resolve_sparsity_generator(::Nothing, ::Val{:NN}) = :NN
resolve_sparsity_generator(::Nothing, ::Val{:HNSW}) = :HNSW

resolve_sparsity_generator(::Nothing, ::Val{:USERGIVEN}) = :NN
resolve_sparsity_generator(::AbstractVector, sym::Val{<:Symbol}) = error("Unsupported Sparsity pattern type: $(sym.val)")

####################################################
# is_csc_format   
####################################################

function is_csc_format(iVecchiaMLE::VecchiaMLEInput)::Bool

    iVecchiaMLE.sparsitygen != :USERGIVEN && return true
    
    rowsL = iVecchiaMLE.rowsL
    colptrL = iVecchiaMLE.colptrL
    n = iVecchiaMLE.n
    
    length(colptrL) != n + 1 && return false

    # colptr should be non-decreasing
    any(diff(colptrL) .< 0) && return false

    # indices in range
    !all(1 .<= rowsL .<= n) && return false

    # check row ordering within each column
    for col in 1:n
        start_idx = colptrL[col]
        end_idx = colptrL[col+1]-1
        !issorted(rowsL[start_idx:end_idx]) && return false
    end

    return true
end

####################################################
# nn_to_csc   
####################################################

"""
    rows, colptr = nn_to_csc(sparmat::Matrix{Float64})

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
* `colptr`: A vector of incides which determine where new columns start. 

"""
function nn_to_csc(sparmat::Matrix{Int})::Tuple{Vector{Int}, Vector{Int}}
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
        rows[(1:len).+idx] .= view(spari, 1:len)
        idx += len

    end
    counts .= cumsum(counts)
    view(colptr, 2:n+1) .+= counts

    return rows, colptr
end
