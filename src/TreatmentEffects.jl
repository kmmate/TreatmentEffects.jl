module TreatmentEffects

export
# ExogenousParticipationModel
	ExogenousParticipationModel,
	ate_estimator,
	bootstrap_distribution

include("modeltypes.jl")
include("exogenous_participation.jl")
end