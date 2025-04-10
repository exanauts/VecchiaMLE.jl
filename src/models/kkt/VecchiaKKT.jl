using CUDA.CUSOLVER
#=
    Custom KKT System for the Vecchia optimiation problem. 
    TODO: Update KKT System function?
=#


struct VecchiaKKTSystem{T, VT, MT<:AbstractMatrix{T}, QN<:MadNLP.AbstractHessian{T, VT}, LS} <: MadNLP.AbstractCondensedKKTSystem{T, VT, MT, QN}

    # number of nonzeros in L
    p::Int
    
    # number of constraints
    n::Int

    # number of nonzeros in hessian. Not p+n!
    nnzh::Int

    # Diagonal Matrices (Stored as vectors)
    # Diagonal matrix NB⁻¹Nᵀ
    NBNT::VT

    # Vector of Vectors which holds s_j
    S::Vector{VT}

    # holds sizes of s_j ~> side length of each sub-block in hessian
    # cumsum format
    m::Vector{Int}

    # Jacobian values
    jac::VT
    # Hessian Values
    hess::VT

    # Scrap Buffer 
    buffer::VT

    linear_solver::LS

    # constraint vector
    c::VT 
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
    cb::MadNLP.SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
    ) where {T,VT}

    nlp = cb.nlp
    n = nlp.cache.n
    p = get_nvar(nlp) - n

    
    # Get S, need potrf_batch and potrs_batch.
    # How do I do this?
    B_cpy = copy(nlp.cache.B)

    # port m to CPU right now, i don't know exactly how to get this?    
    m = zeros(length(nlp.cache.m))
    copyto!(m, nlp.cache.m)

    # What is m?
    println("m\n", m)

    S = [CuVector{T}([T(i == 1) for i in 1:m[j]]) for j in eachindex(m)]
    println("S\n", S)
    
    println("length S: ", length(S))
    println("lengths: ", [length(s) for s in S])
    println("length B_cpy", length(B_cpy))
    println("lengths:", [size(B) for B in B_cpy])
    CUDA.CUSOLVER.potrfBatched!('U', B_cpy)
    CUDA.CUSOLVER.potrsBatched!('U', B_cpy, S)
    
    CUDA.@allowscalar NBNT = CuVector{T}([S[j][1] for j in eachindex(m)])
    

    # NOTE: MadNLP automatically updates these values  
    hess = CUDA.zeros(T, p+n)
    jac = CUDA.zeros(T, 2*n)

    # Setting the buffer avoids edge cases for small k values 
    buffer = CUDA.zeros(T, max(p+n, 3*n))
    
    # Query for constraint vector
    c = # ???

    return VecchiaKKTSystem{T, VT, AbstractMatrix{T}, MadNLP.AbstractHessian{T, VT}, typeof(linear_solver)}(
        p,
        n,
        p+n,
        NBNT, 
        S, 
        m, 
        jac,
        hess,
        buffer,
        linear_solver,
        c
    )
end

#=
    Things to do:
        1. Solve Δz
        2. Solve Δλ
        3. Solve Δy
        4. return true

        w = (Δy, Δz, Δλ)
        Δy is length kkt.p
        Δz is length kkt.n
        Δλ is length kkt.n
        ~> w is length kkt.p + 2*kkt.n  
=#
function MadNLP.solve!(kkt::VecchiaKKTSystem, w::MadNLP.AbstractKKTVector)

    # Hold here for the moment. 
    Λ = view(kkt.jac, (1:kkt.n).+kkt.p)
    D = view(kkt.hess, (1:kkt.n).+kkt.p) ./ Λ

    # get buffer
    buffer = kkt.buffer

    # Solve for Δz
    # RHS ~> save to buffer
    buffer_dz_ptr = view(buffer, 1:p)
    view(buffer_dz_ptr, 1:kkt.n) .= view(w, (1:kkt.n).+kkt.p)
    
    # TODO: MAYBE WRONG!!!
    # TODO: This is for sure wrong, since the subviews are over the colptr, not just m either. 
    subviews = [view(buffer_dz_ptr, kkt.S[i] : (kkt.S[i+1] - 1)) for i in eachindex(kkt.S)]
    view(buffer_dz_ptr, 1:length(kkt.S)) .= dot.(kkt.S, subviews)
    
    buffer_dz_ptr = view(buffer, 1:kkt.n)
    buffer_dz_ptr .= kkt.c .- buffer_dz_ptr # TODO: kkt.c won't work. c is stored in w (where?), wait for francois.
    
    # Need product for later, save off to other part of buffer 
    buffer_cmNBgyL = view(buffer, (1:kkt.n).+2*kkt.n)
    buffer_cmNBgyL .= copy(buffer_dz_ptr)
        
    buffer_dz_ptr .= buffer_dz_ptr ./ NBNT
    buffer_dz_ptr .= kkt.grad_z_Lagrange - buffer_dz_ptr

    # building matrix ΛD + D(NB⁻¹Nᵀ)⁻¹D
    # Should be same size as grad_z_Lagrange
    buffer_DNBNTD_ptr = view(buffer, (1:kkt.n).+kkt.n)

    buffer_DNBNTD_ptr .= D ./ kkt.NBNT
    buffer_DNBNTD_ptr ./= D
    buffer_DNBNTD_ptr .+= view(kkt.hess, kkt.nnzh-kkt.n:kkt.nnzh) # View is on ΛD in Hessian of Lagrangian

    # Now solve for Δz. After Δy in w.
    view(w, (1:kkt.n).+kkt.p) .= - buffer_dz_ptr ./ buffer_DNBNTD_ptr 

    # buffer can now be overwritten 
    # solve for Δλ
    # same size as buffer_dz_ptr! 
    buffer_dlm_ptr = view(buffer, 1:kkt.n)
    buffer_dlm_ptr .= D .* view(w, (1:kkt.n).+kkt.p)
    buffer_dlm_ptr .+= buffer_cmNBgyL
    
    view(w, (1:kkt.n).+(kkt.n+kkt.p)) .= buffer_dlm_ptr ./ kkt.NBNT

    # buffer can now be overwritten
    # solve for Δy
    buffer_dy_ptr = view(buffer, 1:kkt.p)

    # view onto diagonal to add Δλ entries
    # Note negative (of the Nᵀ) cancels out with the other negative
    buffer_diag_y = view(buffer_dy_ptr, kkt.m)
    buffer_diag_y .+= view(w, (1:kkt.n).+(kkt.n+kkt.p))
    # That SHOULD be the RHS of Δy

    # The S_j's are the nonzero elements of B⁻¹. 
    # Fill in Δy.
    view(w, 1:kkt.p) .= dot.(kkt.S[1:length(kkt.m)], Ref(buffer_diag_y))

    return true
end
