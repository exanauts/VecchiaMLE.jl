module VecchiaMLENLPModelsKnitroExt

import VecchiaMLE
import NLPModelsKnitro

function vecchia_solver(::Val{:knitro}, args...; kwargs...)
	NLPModelsKnitro.knitro(args...; kwargs...)
end

end
