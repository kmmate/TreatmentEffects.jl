module TreatmentEffects

export
# ExogenousParticipationModel
	ExogenousParticipationModel,
	ate_estimator

include("modeltypes.jl")
include("exogenous_participation.jl")
end