using CUSOLVER

struct VecchiaKKTSystem{T, VT, MT, QN} <: MadNLP.AbstractCondensedKKTSystem{T, VT, MT, QN}

    # Diagonal Matrices (Stored as vectors)
    # Diagonal matrix NB⁻¹Nᵀ
    NBNT::VT
    # Diagonal matrix D
    D::VT
    # Diagonal Matrix Λ
    Λ::VT

    # gradient of Lagrangian (wrt y)
    grad_y_Lagrange::VT
    # gradient of Lagrangian (wrt z)
    grad_z_Lagrange::VT

    # Vector of constraints
    c::VT

    # Vector of Vectors which holds s_j
    S::Vector{VT}

    # holds sizes of s_j ~> side length of each sub-block in hessian
    # cumsum format
    m::Vector{Int}

    # Scrap Buffer 
    buffer::VT
end

#=
    TODO: Fill out
    Things to do:
        1. Query MadNLPSolver for c, grad_y_Lagrange, grad_z_Lagrange, x, lagrange multipliers
        2. Allocate stuff (buffer)
        3. Calculate NBNT, S, Λ, D.
        4. Create VecchiaKKT
        5. return
 =#

function MadNLP.create_kkt_system(
    ::Type{VecchiaKKTSystem},
    cb::MadNLP.AbstractCallback{T, VT},
    ind_cons, 
    linear_solver;
    opt_linear_solver=MadNLP.default_options(linear_solver)
    ) where {T, VT}

    nlp = cb.nlp
    n = nlp.cache.n
    p = get_nvar(nlp) - n
    grad_y_Lagrange = view(nlp.gradient, 1:p)
    grad_z_Lagrange = view(nlp.gradient, p+1:p+n)

    c = nlp.constraints
    m = nlp.cache.n

    # Get S, need potrf_batch and potrs_batch.
    # How do I do this?
    B_cpy = copy(nlp.cache.B)
    S = [CuVector{T}(m[j]) for j in eachindex(m)]
    CUDA.CUSOLVER.potrfBatched!('U', B_cpy)
    CUDA.CUSOLVER.potrsBatched!('U', B_cpy, S)

    # Next need NBNT, D, Λ, buffer. I don't know how to get these!
    NBNT = [S[j][j] for j in eachindex(m)]
    D = [(nlp.values)[m[i]] for i in eachindex(m)]
    Λ = nlp.lagrange_multipliers
    buffer = zeros(p+n)

    return VecchiaKKTSystem(
        NBNT, 
        D, 
        Λ, 
        grad_y_Lagrange, 
        grad_z_Lagrange, 
        c, 
        S, 
        m, 
        buffer
    )
end

#=
    TODO: Fill out
    Things to do:
        1. Solve Δz
        2. Solve Δλ
        3. Solve Δy
        4. return true

        DON'T KNOW WHAT w SHOULD HOLD! Is it the result?
        w = (Δy, Δz, Δλ)? 
=#
function MadNLP.solve!(kkt::VecchiaKKT, w::MadNLP.AbstractKKTVector)

    # get buffer
    buffer = kkt.buffer

    # Solve for Δz
    # RHS ~> save to buffer
    buffer_dz_ptr = view(buffer, 1:length(kkt.grad_y_Lagrange))
    buffer_dz_ptr .= kkt.grad_y_Lagrange
    
    # Maybe be able to remove for loop, do broadcasting but idk
    # Note that buffer_dz_ptr "shrinks", i.e. relevant stuff is now in first n indices 
    # TODO: MAYBE WRONG!!!
    for i in eachindex(kkt.S)
        buffer_dz_ptr[i] = dot(kkt.S[i][:], view(buffer_dz_ptr, kkt.S[i]:kkt.S[i+1] -1))
    end
    buffer_dz_ptr = view(buffer, 1:length(grad_z_Lagrange))
    buffer_dz_ptr .= kkt.c .- buffer_dz_ptr
    
    # Need product for later, save off to other part of buffer 
    buffer_cmNBgyL = view(buffer, 2*length(kkt.grad_z_Lagrange)+1 : 3*length(kkt.grad_z_Lagrange))
    buffer_cmNBgyL .= copy(buffer_dz_ptr)
    
    
    buffer_dz_ptr .= buffer_dz_ptr ./ NBNT
    buffer_dz_ptr .= kkt.grad_z_Lagrange - buffer_dz_ptr

    # building matrix ΛD + D(NB⁻¹Nᵀ)⁻¹D
    # Should be same size as grad_z_Lagrange
    buffer_DNBNTD_ptr = view(buffer, length(kkt.grad_z_Lagrange)+1:2*length(kkt.grad_z_Lagrange))

    buffer_DNBNTD_ptr .= kkt.D ./ kkt.NBNT
    buffer_DNBNTD_ptr .= buffer_DNBNTD_ptr ./ kkt.D
    buffer_DNBNTD_ptr .+= (kkt.Λ .* kkt.D)             # Parenthesis because I don't know the precedence

    # Now solve for Δz
    Δz = copy(- buffer_dz_ptr ./ buffer_DNBNTD_ptr)
    
    # buffer can now be overwritten 
    # solve for Δλ
    # same size as buffer_dz_ptr! 
    buffer_dlm_ptr = view(buffer, 1:length(kkt.grad_z_Lagrange))
    buffer_dlm_ptr .= kkt.D .* Δz
    buffer_dlm_ptr .+= buffer_cmNBgyL
    Δλ = copy(buffer_dlm_ptr ./ kkt.NBNT)

    # buffer can now be overwritten
    # solve for Δy
    buffer_dy_ptr = view(buffer, 1:length(kkt.grad_y_Lagrange))
    buffer_dy_ptr = -kkt.grad_y_Lagrange
    # view onto diagonal to add Δλ entries
    buffer_diag_y = view(buffer_dy_ptr, kkt.m)
    buffer_diag_y .= Δλ         # Note negative (of the Nᵀ) cancels out with the other negative 
    # That SHOULD for the RHS of Δy

    
    # The S_j's are the nonzero elements of B⁻¹. 
    Δy = zeros(kkt.grad_y_Lagrange)
    for i in eachindex(kkt.m)
        Δy[i] = dot(kkt.S[i][:], buffer_diag_y)
    end

    return true
end