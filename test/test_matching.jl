#=
Tests for matching.jl


Remarks
-------



=#

# testing ate_matchingestimator
@testset "ate_matchingestimator" begin
	# read example dataset
	#datapath = normpath(joinpath(dirname(pathof(TreatmentEffects)), "..", "test\\data\\mathcing"))
	#y = CSV.read(joinpath(datapath, "y.csv"), header=false)
	#d = CSV.read(joinpath(datapath, "d.csv"), header=false)
	#x = CSV.read(joinpath(datapath, "X.csv"), header=false)
	y_raw = Array(CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\y.csv", header=false)[:1])
	d_raw = Array(CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\d.csv", header=false)[:1])
	x_raw = convert(Matrix, CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\X.csv", header=false, delim=';'))
	y, d, x = drop_missing(y_raw, d_raw, x_raw)
	# set up model
	mam = MatchingModel(y, d, x')
	println(typeof(mam))
	# estimate ATE
	numberof_neighbours = 1
	matching_method = :covariates
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	println("ATE_hat = ", ate_hat)
	@test isapprox(ate_hat, -200, atol=10) == true
end