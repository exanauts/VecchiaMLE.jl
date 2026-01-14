module VecchiaMLEVecchiaExt

  using VecchiaMLE, Vecchia, StaticArrays

  function VecchiaMLE.VecchiaModel(pts::Vector{SVector{D,Float64}},
                                   data::Matrix{Float64},
                                   ordering, conditioning;
                                   lvar_diag=fill(1e-10, length(pts)),
                                   uvar_diag=fill(1e10, length(pts)),
                                   lambda=0.0) where{D}
    va = VecchiaApproximation(pts, nothing, data; ordering=ordering,
                              conditioning=conditioning)
    VecchiaModel(va; lvar_diag=lvar_diag, uvar_diag=uvar_diag, lambda=lambda)
  end

  function VecchiaMLE.VecchiaModel(va::VecchiaApproximation;
                                   lvar_diag=fill(1e-10, length(pts)),
                                   uvar_diag=fill(1e10, length(pts)),
                                   lambda=0.0)
    (condsets, perm) = (va.condix, va.perm)
    n   = length(condsets)
    nnz = sum(length, condsets) + length(condsets)
    IJ  = Vector{Tuple{Int64, Int64}}()
    sizehint!(IJ, nnz)
    for (j, cj) in enumerate(condsets)
      foreach(k->push!(IJ, (k, j)), cj)
      push!(IJ, (j,j))
    end
    (I, J) = (getindex.(IJ, 1), getindex.(IJ, 2))
    (perm, VecchiaModel(I, J, permutedims(va.data);
                        uplo=:U, lvar_diag=lvar_diag, 
                        uvar_diag=uvar_diag, lambda=lambda))
  end

end
