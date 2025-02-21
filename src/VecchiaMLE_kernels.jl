@kernel function vecchia_mul_kernel!(y, @Const(hess_obj_vals), @Const(x), @Const(m), @Const(n), @Const(colptrL))
    index = @index(Global)
    if index <= n
        # Compute the starting position for the current block using the column pointer
        pos = colptrL[index]
        mj = m[index]

        # Perform the matrix-vector multiplication for the current symmetric block
        for i in 1:mj
            # Add the contribution of the diagonal element A[i,i]
            diag_index = (i * (i - 1)) ÷ 2 + i
            y[pos-1+i] += hess_obj_vals[diag_index] * x[pos-1+i]
            # Loop over off-diagonal elements in row i (for j < i)
            for j in 1:i-1
                idx = (i * (i - 1)) ÷ 2 + j  # Compute the index of A[i,j] in the compact vector
                a = hess_obj_vals[idx]
                y[pos-1+i] += a * x[pos-1+j]
                y[pos-1+j] += a * x[pos-1+i]  # due to symmetry A[i,j] = A[j,i]
            end
        end
    end
    nothing
end

function vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, hess_obj_vals::CuVector{T}, x::CuVector{T}, n::Int, m::Vector{Int}, colptrL::CuVector{T}) where T <: AbstractFloat
    # Reset the vector y
    fill!(y, zero(T))

    # Launch the kernel
    backend = KA.get_backend(y)
    vecchia_mul_kernel!(backend)(y, hess_obj_vals, x, m, n, colptrL, ndrange=n)
    KA.synchronize(backend)
    return y
end

function vecchia_mul!(y::Vector{T}, B::Vector{Matrix{T}}, hess_obj_vals::Vector{T}, x::Vector{T}, n::Int, m::Vector{Int}, colptrL::Vector{T}) where T <: AbstractFloat
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

# We want to replace this function by a kernel implemented with KernelAbstractions.jl
function vecchia_build_B!(B::Vector{Matrix{T}}, samples::CuMatrix{T}, rowsL::Vector{Int}, colptrL::Vector{Int}, hess_obj_vals::Vector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
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
end

function vecchia_build_B!(B::Vector{Matrix{T}}, samples::Matrix{T}, rowsL::Vector{Int}, colptrL::Vector{Int}, hess_obj_vals::Vector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
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
end
