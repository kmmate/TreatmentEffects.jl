module TreatmentEffects

export
	# data cleaning
	drop_missing,
	# Exogenous Participation
	ExogenousParticipationModel,
	ate_estimator,
	bootstrap_distribution,
	bootstrap_htest,
	# Matching
	MatchingModel,
	ate_matchingestimator,
	att_matchingestimator,
	ate_blockingestimator,
	# nonparametric
	uniform_kernel,
	triangular_kernel,
	gaussian_kernel,
	epanechnikov_kernel,
	localpoly_regression,
	polynomial_features,
	# other estimators
	predict_propscore,
	# rdd
	RDDModel,
	rdd_sharpestimator,
	# Paired Interference
	PairedInterferenceModel,
	late_estimator,
	bootstrap_distribution


include("data_cleaning.jl")
include("modeltypes.jl")
include("estimators.jl")
include("nonparametrics/kernels.jl")
include("nonparametrics/np_regression.jl")
include("exogenous_participation.jl")
include("matching.jl")
include("rdd.jl")
include("paired_interference.jl")
end
