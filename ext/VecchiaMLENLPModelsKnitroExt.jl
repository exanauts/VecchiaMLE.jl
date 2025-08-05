module VecchiaMLENLPModelsKnitroExt

import VecchiaMLE
import NLPModelsKnitro

function vecchia_solver(solver::Val{:knitro}, args...; kwargs...)
	NLPModelsKnitro.knitro(args...; kwargs...)
end

end
