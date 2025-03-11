#=
    VECCHIA_MUL
    Front-end
    kernel
    CPU implementation
=#

# Front-end
function vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, hess_obj_vals::CuVector{T},
    x::CuVector{T}, n::Int, m::CuVector{Int}, offsets::CuVector{Int}) where T <: AbstractFloat
    # Reset the vector y
    fill!(y, zero(T))

    # Launch the kernel
    backend = KA.get_backend(y)
    kernel = vecchia_mul_kernel!(backend)
    kernel(y, hess_obj_vals, x, m, offsets, ndrange=n)
    KA.synchronize(backend)
    return y
end


# kernel
@kernel function vecchia_mul_kernel!(y, @Const(hess_obj_vals), @Const(x), @Const(m), @Const(offsets))
    index = @index(Global)
    offset = offsets[index]
    mj = m[index]
    pos2 = 0
    for i = 1:index-1
        pos2 += m[i] * (m[i] + 1) ÷ 2
    end

    # Perform the matrix-vector multiplication for the current symmetric block
    for j in 1:mj
        idx1 = (j - 1) * (mj + 1) - j * (j-1) ÷ 2
        for i in j:mj
            idx2 = idx1 + (i - j + 1)
            val = hess_obj_vals[pos2+idx2]

            # Diagonal element contributes only once
            if i == j
                y[offset+i] += val * x[offset+j]
            else
                y[offset+i] += val * x[offset+j]
                y[offset+j] += val * x[offset+i]  # due to symmetry A[i,j] = A[j,i]
            end
        end
    end
    nothing
end

# CPU implementation
function vecchia_mul!(y::Vector{T}, B::Vector{Matrix{T}}, hess_obj_vals::Vector{T},
                      x::Vector{T}, n::Int, m::Vector{Int}, offsets::Vector{Int}) where T <: AbstractFloat
    pos = 0
    for j = 1:n
        Bj = B[j]
        yj = view(y, pos+1:pos+m[j])
        xj = view(x, pos+1:pos+m[j])
        mul!(yj, Bj, xj)
        pos = pos + m[j]
    end
    return y
end

#= 
    VECCHIA_BUILD_B
    Front-end 
    kernel
    CPU implementation
=#

# Front-end
function vecchia_build_B!(B::Vector{<:CuMatrix{T}}, samples::CuMatrix{T}, rowsL::CuVector{Int},
    colptrL::CuVector{Int}, hess_obj_vals::CuVector{T}, n::Int, m::CuVector{Int}) where T <: AbstractFloat
    # Launch the kernel
    backend = KA.get_backend(samples)
    r = size(samples, 1)
    kernel = vecchia_build_B_kernel!(backend)
    kernel(hess_obj_vals, samples, rowsL, colptrL, m, r, ndrange=n)
    KA.synchronize(backend)
    return nothing
end

# kernel
@kernel function vecchia_build_B_kernel!(hess_obj_vals, @Const(samples), @Const(rowsL), @Const(colptrL), @Const(m), @Const(r))
    index = @index(Global)
    pos = colptrL[index]
    mj = m[index]
    pos2 = 0
    for i = 1:index-1
        pos2 += m[i] * (m[i] + 1) ÷ 2
    end

    k = 0
    for s in 1:mj
        for t in s:mj
            acc = 0.0
            for i = 1:r
                acc += samples[i, rowsL[pos+t-1]] * samples[i, rowsL[pos+s-1]]
            end
            k = k + 1
            hess_obj_vals[pos2+k] = acc
        end
    end
    nothing
end

# CPU implementation
function vecchia_build_B!(B::Vector{Matrix{T}}, samples::Matrix{T}, rowsL::Vector{Int},
                          colptrL::Vector{Int}, hess_obj_vals::Vector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
    pos = 0
    for j in 1:n
        for s in 1:m[j]
            for t in 1:m[j]
                vt = view(samples, :, rowsL[colptrL[j] + t - 1])
                vs = view(samples, :, rowsL[colptrL[j] + s - 1])
                B[j][t, s] = dot(vt, vs)

                # Lower triangular part of the block Bⱼ
                if s ≤ t
                    pos = pos + 1
                    hess_obj_vals[pos] = B[j][t, s]
                end
            end
        end
    end
    return nothing
end

#=
    VECCHIA_GENERATE_HESS_TRI_STRUCTURE
    Front-end
    kernel
    CPU implementation
=#

# Front-end
function vecchia_generate_hess_tri_structure!(nnzh::Int, n::Int, colptr_diff::CuVector{Int}, 
    hrows::CuVector{Int}, hcols::CuVector{Int})

    # reset hrows, hcols
    fill!(hrows, one(Int))
    fill!(hcols, one(Int))

    # launch the kernel
    backend = KA.get_backend(hrows)
    kernel = vecchia_generate_hess_tri_structure_kernel!(backend)

    f(x) = (x * (x+1)) ÷ 2

    # NOTE: Might be a race condition here. Solution is to store them in the first indices of each thread. 
    view(hrows, 2:n) .+= cumsum(f.(view(colptr_diff, 1:n-1)))
    view(hcols, 2:n) .+= cumsum(view(colptr_diff, 1:n-1))

    kernel(nnzh, n, colptr_diff, hrows, hcols, hrows, hcols, ndrange = n)

    KA.synchronize(backend)

    return nothing
end

# kernel
@kernel function vecchia_generate_hess_tri_structure_kernel!(
    @Const(nnzh), @Const(n), @Const(colptr_diff), @Const(carry_offsets), @Const(idx_offsets),
    hrows, hcols
)
    thread_idx = @index(Global) # in 1:n
    m = colptr_diff[thread_idx]
    carry = carry_offsets[thread_idx]
    idx = idx_offsets[thread_idx]

    for j in 1:m
        for k in carry:m - j + carry
            hrows[k] = (idx + j - 1) + (k - carry)
            hcols[k] = idx + j - 1
        end
        carry += m - j + 1
    end

    # fill one index of the tail for each thread
    @inbounds hrows[nnzh-n + thread_idx] = hrows[nnzh-n] + thread_idx
    @inbounds hcols[nnzh-n + thread_idx] = hrows[nnzh-n] + thread_idx
end



# CPU implementation
function vecchia_generate_hess_tri_structure!(nnzh::Int, n::Int, colptr_diff::Vector{Int}, 
    hrows::Vector{Int}, hcols::Vector{Int}) 
    
    carry = 1
    idx = 1
    for i in 1:n
        m = colptr_diff[i]
            for j in 1:m
                view(hrows, (0:(m-j)).+carry) .= (j:m).+(idx-1)
                fill!(view(hcols, carry:carry+m-j), idx + j - 1)
                carry += m - j + 1
            end
        idx += m
    end

    #Then need the diagonal tail
    idx_to = idx + nnzh - carry
    view(hrows, carry:nnzh) .= idx:idx_to
    view(hcols, carry:nnzh) .= idx:idx_to

    return hrows, hcols
end
