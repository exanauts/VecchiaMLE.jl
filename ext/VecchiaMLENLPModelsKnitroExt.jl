module VecchiaMLENLPModelsKnitroExt

import VecchiaMLE
import NLPModelsKnitro

function VecchiaMLE.vecchia_solver(::Val{:knitro}, args...; kwargs...)
	NLPModelsKnitro.knitro(args...; kwargs...)
end

end
