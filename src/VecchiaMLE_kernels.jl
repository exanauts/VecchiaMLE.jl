# @kernel inbounds=true function vecchia_mul_kernel!(y, @Const(hess_obj_vals), @Const(x), @Const(m), @Const(n), @Const(colptrL), @Const(offsets))
#     index = @index(Global)

#     pos = colptrL[index]
#     offset = offsets[index]
#     mj = m[index]

#     # Perform the matrix-vector multiplication for the current symmetric block
#     for j in 1:mj
#         idx1 = j * (j-1) ÷ 2
#         for i in j:mj
#             idx2 = idx1 + (i - j + 1)
#             val = hess_obj_vals[pos-1+idx2]

#             # Diagonal element contributes only once
#             if i == j
#                 y[offset+i] += val * x[offset+j]
#             else
#                 y[offset+i] += val * x[offset+j]
#                 y[offset+j] += val * x[offset+i]  # due to symmetry A[i,j] = A[j,i]
#             end
#         end
#     end
#     nothing
# end

# function vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, hess_obj_vals::CuVector{T}, x::CuVector{T}, n::Int, m::Vector{Int}, colptrL::CuVector{Int}) where T <: AbstractFloat
#     # Reset the vector y
#     fill!(y, zero(T))

#     # Precompute offsets for all blocks <-- should be in VecchiaCache
#     offsets = cumsum([0; m[1:end-1]]) |> CuVector{Int}

#     # Launch the kernel
#     backend = KA.get_backend(y)
#     vecchia_mul_kernel!(backend)(y, hess_obj_vals, x, CuVector(m), n, colptrL, offsets, ndrange=n)
#     KA.synchronize(backend)
#     return y
# end

function vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, hess_obj_vals::CuVector{T}, x::CuVector{T}, n::Int, m::Vector{Int}, colptrL::CuVector{Int}) where T <: AbstractFloat
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

function vecchia_mul!(y::Vector{T}, B::Vector{Matrix{T}}, hess_obj_vals::Vector{T}, x::Vector{T}, n::Int, m::Vector{Int}, colptrL::Vector{Int}) where T <: AbstractFloat
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

@kernel function vecchia_build_B_kernel!(hess_obj_vals, @Const(samples), @Const(rowsL), @Const(colptrL),
                                         @Const(m), @Const(n), @Const(r))
    index = @index(Global)

    pos = colptrL[index]
    mj = m[index]

    k = 0
    for s in 1:mj
        for t in s:mj
            acc = 0.0
            for i = 1:r
                acc += samples[i, rowsL[pos + t - 1]] * samples[i, rowsL[pos + s - 1]]
            end
            hess_obj_vals[pos+k] = acc
            k = k + 1
        end
    end
    nothing
end

function vecchia_build_B!(B::Vector{Matrix{T}}, samples::CuMatrix{T}, rowsL::Vector{Int}, colptrL::Vector{Int}, hess_obj_vals::CuVector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
    for j in 1:n
        for s in 1:m[j]
            for t in 1:m[j]
                vt = view(samples, :, rowsL[colptrL[j] + t - 1])
                vs = view(samples, :, rowsL[colptrL[j] + s - 1])
                B[j][t, s] = dot(vt, vs)
            end
        end
    end

    # Launch the kernel
    backend = KA.get_backend(samples)
    r = size(samples, 1)
    vecchia_build_B_kernel!(backend)(hess_obj_vals, samples, CuVector(rowsL), CuVector(colptrL), CuVector(m), n, r, ndrange=n)
    KA.synchronize(backend)
    return nothing
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
    return nothing
end
