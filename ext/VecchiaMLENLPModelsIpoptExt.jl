module VecchiaMLENLPModelsIpoptExt

import VecchiaMLE
import NLPModelsIpopt

function vecchia_solver(::Val{:ipopt}, args...; kwargs...)
	NLPModelsIpopt.ipopt(args...; kwargs...)
end

end
