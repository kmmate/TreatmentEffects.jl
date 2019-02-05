#=
Tests for matching.jl


Remarks
-------



=#

# testing ate_matchingestimator
@testset "ate_matchingestimator" begin
	#		Real data test 		  #
	#-----------------------------#
	
	# read example dataset
	# subsampling
	n_subsample = 25_000
	subsample_idx = rand(1:60_000, n_subsample)
	#datapath = normpath(joinpath(dirname(pathof(TreatmentEffects)), "..", "test\\data\\mathcing"))
	#y = CSV.read(joinpath(datapath, "y.csv"), header=false)
	#d = CSV.read(joinpath(datapath, "d.csv"), header=false)
	#x = CSV.read(joinpath(datapath, "X.csv"), header=false)
	y_raw = Array(CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\y.csv", header=false)[:1])[subsample_idx]
	d_raw = Array(CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\d.csv", header=false)[:1])[subsample_idx]
	x_raw = Array(convert(Matrix, CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\X.csv", header=false, delim=';')))[subsample_idx, :]
	y, d, x = drop_missing(y_raw, d_raw, x_raw)
	# set up model
	mam = MatchingModel(y, d, x)
	# estimate ATE
	#	---	test mathcing on covariates
	numberof_neighbours = 1
	matching_method = :covariates
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	println("ATE_hat = ", ate_hat)
	@test isapprox(ate_hat, -200, atol=50) == true
	#	---	test mathcing on covariates
	numberof_neighbours = 1
	matching_method = :propscore_logit
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	println("ATE_hat = ", ate_hat)
	@test isapprox(ate_hat, -200, atol=50) == true
end