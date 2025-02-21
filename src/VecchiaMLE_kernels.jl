@kernel function vecchia_mul_kernel!(y, @Const(B), @Const(x), @Const(m), @Const(n), @Const(offsets))
    index = @index(Global)
    if index <= n
        # Compute the starting position for the current block
        pos = offsets[index]
        mj = m[index]
        Bj = B[index]

        # Perform the matrix-vector multiplication for B[index]
        for s in 1:mj
            for t in 1:mj
                y[pos+s] += Bj[s,t] * x[pos+t]
            end
        end
    end
    nothing
end

function vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, x::CuVector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
    # Precompute offsets for all blocks <-- should be in VecchiaCache
    offsets = cumsum([0; m[1:end-1]])

    # Reset the vector y
    fill!(y, zero(T))

    # Launch the kernel
    backend = KA.get_backend(y)
    vecchia_mul_kernel!(backend)(y, B, x, m, n, offsets, ndrange=n)
    KA.synchronize(backend)
    return y
end

function vecchia_mul!(y::Vector{T}, B::Vector{Matrix{T}}, x::Vector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
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
