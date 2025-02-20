# We want to replace this function by a kernel implemented with KernelAbstractions.jl
function vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, x::CuVector{T}, n::Int, m::Vector{Int}) where T <: AbstractFloat
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
