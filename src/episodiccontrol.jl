using TabularReinforcementLearning, DataStructures.PriorityQueue

struct EpisodicControl <: TabularReinforcementLearning.AbstractReinforcementLearner
	γ::Float64
	Q::Array{Float64, 2}
end
EpisodicControl(; ns = 10, na = 4, γ = 1., initvalue = Inf64) =
	EpisodicLearner(EpisodicControl(γ, zeros(na, ns) + initvalue))

import TabularReinforcementLearning.update!, TabularReinforcementLearning.act
function update!(::EpisodicLearner, learner::EpisodicControl, rewards, states, 
				 actions, isterminal)
	if isterminal
		G = rewards[end]
		for t in length(rewards)-1:-1:1
			G = learner.γ * G + rewards[t]
			if G > learner.Q[actions[t], states[t]] || 
					learner.Q[actions[t], states[t]] == Inf64
				learner.Q[actions[t], states[t]] = G
			end
		end
	else
		if learner.Q[actions[end], states[end]] == Inf64
			learner.Q[actions[end], states[end]] = -Inf64
		end
	end
end
act(learner::EpisodicControl, policy, state) = act(policy, learner.Q[:, state])

mutable struct ModelReset <: TabularReinforcementLearning.AbstractCallback 
	counter::Int64
end
ModelReset() = ModelReset(0)
import TabularReinforcementLearning.callback!
function callback!(c::ModelReset, learner::SmallBackups, p, r, a, s, isterminal)
	c.counter += 1
	if isterminal
		learner.maxcount = c.counter
		TabularReinforcementLearning.processqueue!(learner)
		learner.Nsa .= 0
		learner.Ns1a0s0 = [Dict{Tuple{Int64, Int64}, Int64}() for _ in
						   1:length(learner.Ns1a0s0)]
		learner.queue = PriorityQueue(Int64[], Float64[], Base.Order.Reverse)
		c.counter = 0
		learner.maxcount = 0
	end
end
