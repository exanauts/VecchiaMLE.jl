# These functions are only for internal use, and are not intended for the user to call them!


####################################################
#               validate_input                
####################################################

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
    @assert_in iVecchiaMLE.linear_solver LINEAR_SOLVERS

    # Check package version of  instead
    (!HSL.LIBHSL_isfunctional() && iVecchiaMLE.linear_solver in (:ma27, :ma57)) &&
         error("LIBHSL_isfunctional() returned false. $linear_solver is not available.")

end


####################################################
#               Resolve vecchia_solver                
####################################################
function vecchia_solver(::Val{s}, args...; kwargs...) where {s}
    error("The solver $s is not available.")
end

function vecchia_solver(::Val{:madnlp}, args...; kwargs...)
    madnlp(args...; kwargs...)
end

####################################################
#               Resolve linear_solver                
####################################################
resolve_linear_solver(::Val{:madnlp}, ::Val{:umfpack}) = MadNLPHSL.UmfpackSolver
resolve_linear_solver(::Val{:madnlp}, ::Val{:ma27}) = MadNLPHSL.Ma27Solver
resolve_linear_solver(::Val{:madnlp}, ::Val{:ma57}) = MadNLPHSL.Ma57Solver

resolve_linear_solver(solver::Val{:knitro}, ::Val{:umfpack}) = error("linear_solver for $solver not yet implemented!")
resolve_linear_solver(solver::Val{:knitro}, ::Val{:ma27}) = error("linear_solver for $solver not yet implemented!")
resolve_linear_solver(solver::Val{:knitro}, ::Val{:ma57}) = error("linear_solver for $solver not yet implemented!")


resolve_linear_solver(solver::Val{:ipopt}, ::Val{:ma27}) = "ma27"
resolve_linear_solver(solver::Val{:ipopt}, ::Val{:ma57}) = "ma57"

resolve_linear_solver(solver::Val{<:Symbol}, lin::Val{<:Symbol}) = error("solver $solver does not support linear solver $lin")

####################################################
#                resolve_plevel
####################################################

resolve_plevel(::Val{:madnlp}, plevel::Val{T}) where {T} = error("Unsupported print level $(T) for solver :madnlp.")
resolve_plevel(solver::Val{<:Symbol}, ::Val{T}) where {T} = error("The solver $(solver) does not have defined print level $(T).")


resolve_plevel(::Val{:madnlp}, ::Val{:VTRACE}) = MadNLP.TRACE
resolve_plevel(::Val{:madnlp}, ::Val{:VDEBUG}) = MadNLP.DEBUG
resolve_plevel(::Val{:madnlp}, ::Val{:VINFO})  = MadNLP.INFO
resolve_plevel(::Val{:madnlp}, ::Val{:VWARN})  = MadNLP.WARN
resolve_plevel(::Val{:madnlp}, ::Val{:VERROR}) = MadNLP.ERROR
resolve_plevel(::Val{:madnlp}, ::Val{:VFATAL}) = MadNLP.ERROR


####################################################
#                convert_plevel
####################################################

convert_plevel(::Val{T}) where {T} = error("Unsupported print level $T")

convert_plevel(::Val{:VTRACE}) = :VTRACE
convert_plevel(::Val{1})       = :VTRACE
convert_plevel(::Val{:VDEBUG}) = :VDEBUG
convert_plevel(::Val{2})       = :VDEBUG
convert_plevel(::Val{:VINFO})  = :VINFO
convert_plevel(::Val{3})       = :VINFO
convert_plevel(::Val{:VWARN})  = :VWARN
convert_plevel(::Val{4})       = :VWARN
convert_plevel(::Val{:VERROR}) = :VERROR
convert_plevel(::Val{5})       = :VERROR

####################################################
#              convert_computemode
####################################################

convert_computemode(::Val{arch}) where {arch} = error("Unsupported architecture: $arch")

convert_computemode(::Val{:cpu}) = :cpu
convert_computemode(::Val{1})    = :cpu

convert_computemode(::Val{:gpu}) = :gpu
convert_computemode(::Val{2})    = :gpu


####################################################
# tovector. 
# To convert ptset from Matrix to Vector of Vectors
####################################################

function tovector(A::AbstractMatrix)::AbstractVector
    return size(A, 1) > size(A, 2) ? [row for row in eachrow(A)] :
                                     [col for col in eachcol(A)]
end


####################################################
# check_x0!, check_lvar!, check_uvar!
# Functions to check user given arrays. 
####################################################

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


####################################################
# resolve_ptset
# MD for resolving user given (or not) pset .   
####################################################
resolve_ptset(::Int, ptset) = error("Unsupported ptset type: $(typeof(ptset))")
resolve_ptset(n::Int, ::Nothing) = generate_safe_xyGrid(n)
resolve_ptset(::Int, ptset::AbstractMatrix) = tovector(ptset)
resolve_ptset(::Int, ptset::AbstractVector{<:AbstractVector}) = ptset


####################################################
# resolve_sparistygen
# MD for resolving user given (or not) 
# sparsity pattern.   
####################################################

# Cases where user gives a sparsity pattern (doesn't matter its contents, just as long as its of type AbstractVector{<:Unsigned} )
resolve_sparistygen(::AbstractVector{<:Unsigned}, ::Val{:NN}) = :USERGIVEN
resolve_sparistygen(::AbstractVector{<:Unsigned}, ::Val{:HNSW}) = :USERGIVEN
resolve_sparistygen(::AbstractVector{<:Unsigned}, ::Val{:USERGIVEN}) = :USERGIVEN
resolve_sparistygen(::Nothing, ::Val{:NN}) = :NN
resolve_sparistygen(::Nothing, ::Val{:HNSW}) = :HNSW

# WARN: This should not happen, since this would be detected by validate_input. Just a fallback.
resolve_sparistygen(::Nothing, ::Val{:USERGIVEN}) = :NN
resolve_sparistygen(::AbstractVector, sym::Val{<:Symbol}) = error("Unsupported Sparsity pattern type: $(sym.val)")


####################################################
# is_csc_format   
####################################################

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

####################################################
# nn_to_csc   
####################################################


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

####################################################
# get_vecchia_model   
####################################################

"""
    model = get_vecchia_model(iVecchiaMLE::VecchiaMLEInput)

    Creates and returns a vecchia model based on the VecchiaMLEInput and point grid. 
## Input arguments

* `iVecchiaMLE`: The filled out VecchiaMLEInput struct
## Output arguments

* `model`: The Vecchia model based on the VecchiaMLEInput  
"""
get_vecchia_model(iVecchiaMLE::VecchiaMLEInput)::VecchiaModel =  get_vecchia_model(iVecchiaMLE, Val(iVecchiaMLE.arch))

get_vecchia_model(iVecchiaMLE::VecchiaMLEInput, ::Val{:cpu}) = VecchiaModelCPU(iVecchiaMLE.samples, iVecchiaMLE)
get_vecchia_model(iVecchiaMLE::VecchiaMLEInput, ::Val{:gpu}) = VecchiaModelGPU(iVecchiaMLE.samples, iVecchiaMLE)
