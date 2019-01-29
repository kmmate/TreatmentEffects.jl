module TreatmentEffects

export
	ExogenousParticipationModel,
	ate_estimator,
	bootstrap_distribution,
	bootstrap_htest

include("modeltypes.jl")
include("exogenous_participation.jl")
end