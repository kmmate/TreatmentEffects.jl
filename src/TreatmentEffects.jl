module TreatmentEffects

export ExogenousParticipationModel,
	   ate_estimator

include("modeltypes.jl")
include("exogenous_participation.jl")
end