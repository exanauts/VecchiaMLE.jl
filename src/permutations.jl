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