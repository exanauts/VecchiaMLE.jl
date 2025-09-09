
"""
    rows, colptr = sparsitypattern(
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
* `colptr`: A vector of incides which determine where new columns start. 

"""
function sparsitypattern(::Val{:NN}, iVecchiaMLE::VecchiaMLEInput)
    return sparsitypattern_NN(iVecchiaMLE.ptset, iVecchiaMLE.k, iVecchiaMLE.metric)
end

function sparsitypattern(::Val{:HNSW}, iVecchiaMLE::VecchiaMLEInput)
    return sparsitypattern_HNSW(iVecchiaMLE.ptset, iVecchiaMLE.k, iVecchiaMLE.metric)
end

function sparsitypattern(::Val{:USERGIVEN}, iVecchiaMLE::VecchiaMLEInput)
    return iVecchiaMLE.rowsL, iVecchiaMLE.colptrL
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
function sparsitypattern_NN(data, k, metric::Distances.Metric=Distances.Euclidean())::Tuple{Vector{Int}, Vector{Int}}
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
function sparsitypattern_HNSW(data, k, metric::Distances.Metric=Distances.Euclidean())::Tuple{Vector{Int}, Vector{Int}}
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
