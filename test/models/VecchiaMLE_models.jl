#=
    Uses MLE to obtain the Vecchia approximation of 
    Covariance Matrix
    params=[5.0, 0.05, 2.25, 0.25] for matern matrix.

=#


function VecchiaMLE_models(n, k, samples, Number_of_Samples, Sparsity, estimationString)
    if estimationString == "Row"
        return VecchiaMLE_Row(n, k, samples, Number_of_Samples, Sparsity)
    elseif estimationString == "Matrix_Sparse" 
        return VecchiaMLE_Matrix_Sparse(n, k, samples, Number_of_Samples, Sparsity)
    elseif estimationString == "MadNLPGPU"
        return VecchiaMLE_Row_Right_MadNLPGPU(n, k, samples, Number_of_Samples, Sparsity)
    else
        println("estimationString incorrect. called ", estimationString)
        return nothing
    end
end

#=
    These functions perform the best in their category (by row or matrix). 
=#

function VecchiaMLE_Row(n, k, samples, Number_of_Samples, Sparsity)

    L = spzeros(n^2, n^2)
    model_sum = Vector{AffExpr}(undef, Number_of_Samples)

    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=100))

    @variable(model, ys[1:k+1])
    set_start_value(ys[1], 1.0)

    # rows are independent -> parallelization, but gpu or cpu?
    for row_idx in 1:n^2    
        model_sum = sum(dot(ys[1:length(Sparsity[row_idx])], samples[i, Sparsity[row_idx]]).^2 for i in 1:Number_of_Samples)
        
        @objective(model, Min,
            -Number_of_Samples * (log(ys[1]) - 
            1.0/(2*Number_of_Samples) * model_sum)
        )

        optimize!(model)
        L[row_idx, Sparsity[row_idx]] = value.(ys)[1:length(Sparsity[row_idx])]
    end    
    return tril(L)  
end

#=
Big issue with this is transferring to ExaModel and gpu compilation speeds
kill it.  
=#
function VecchiaMLE_Matrix_Sparse(n, k, samples, Number_of_Samples, xyGrid)

    Sparsity_pattern = SparsityPattern(xyGrid, k)
    model = VecchiaMLE_Matrix_JuMP_Model(n, k, samples, Number_of_Samples, Sparsity_pattern)
    
    exa_model = ExaModel(model; backend = nothing #= CUDABackend() =#) # model_sum not CUDABackend friendly
    sol = madnlp(exa_model, print_level=MadNLP.ERROR).solution
    
    # Can be easily parallelized
    L = spzeros(n^2, n^2)
    carry = 1
    for j in 1:n^2
        L[j, Sparsity_pattern[j]] = sol[carry:carry+length(Sparsity_pattern[j])-1]
        carry += length(Sparsity_pattern[j])
    end

    return tril(L)
end

#=
    HELPER FUNCTIONS
=#

#=
Returns an array which contains iterates of the desired 
indexing pattern. Note the pattern will generate an UPPER
Triangular matrix, where indexing goes by columns. The
reason for this is when we transpose, for L^T x, the indexing 
cleanly iterates on the rows of L^T. 
=#

function get_idx_pattern(n, k, nneighbors)
    Iterate_arr = Vector{Vector{Int}}(undef, n^2) 
    diag_arr = Vector{Int}(undef, n^2)
    diag_arr[1] = 1
    for i in 1:n^2
        it_row = []
        for (j, row) in enumerate(nneighbors)
            if i in row
                push!(it_row, j)
            end
        end
        Iterate_arr[i] = it_row
        if i < n^2 diag_arr[i+1] = diag_arr[i] + length(it_row) end
    end
    return diag_arr, Iterate_arr
end


#=
    Making a JuMP model to be ported to ExaModels -> gpu.
    Hence no Optimizer.
=#
function VecchiaMLE_Matrix_JuMP_Model(n, k, samples, Number_of_Samples, Sparsity_pattern)
    model = JuMP.Model()
    # one vector variable input, just a big vector. 
    @variable(model, ys[1:Int(0.5 * (k+1) * (2*n^2 - k))])

    # Setting initial value for diagonal
    diag_arr = accumulate(+, [1; length.(Sparsity_pattern)])[1:end-1]

    for idx in diag_arr
        set_start_value(ys[idx], 1.0)
    end

    # Precompute the model_sum
    model_sum = QuadExpr()
    cumulative_indices = cumsum(length.(Sparsity_pattern))
    
    for j in 1:n^2
        for i in 1:Number_of_Samples
            start_idx = j == 1 ? 1 : cumulative_indices[j-1] + 1
            end_idx = cumulative_indices[j]
            term = dot(ys[start_idx:end_idx], samples[i, Sparsity_pattern[j]])^2
            i*j == 1 ? model_sum = term : model_sum += term 
        end
    end

    @objective(model, Min,
        -Number_of_Samples * sum(log(ys[x]) for x in diag_arr) + 
        0.5 * model_sum
    )
    return model
end