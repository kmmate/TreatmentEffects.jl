#=
Tests for matching.jl


Remarks
-------



=#

# testing ate_matchingestimator
@testset "ate_matchingestimator" begin
	# read example dataset
	datapath = normpath(joinpath(dirname(pathof(TreatmentEffects)), "..", "test\\data\\mathcing"))
	#y = CSV.read(joinpath(datapath, "y.csv"), header=false)
	#d = CSV.read(joinpath(datapath, "d.csv"), header=false)
	#x = CSV.read(joinpath(datapath, "X.csv"), header=false)
	y_raw = CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\y.csv", header=false)
	d_raw = CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\d.csv", header=false)
	x_raw = CSV.read("C:\\Users\\Máté\\.julia\\packages\\TreatmentEffects\\SM8SJ\\test\\data"*
		"\\matching\\X.csv", header=false)
	df = hcat(y_raw, d_raw, x_raw; makeunique=true)
	# filter out missing obs
	df = df[completecases(df), :]
	println(df[1:10, :])
	# set up model
	mam = MatchingModel(df[:y], df[:d], df[:x]')
	# estimate ATE
	numberof_neighbours = 1
	matching_method = :covariates
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	println("ATE_hat = ", ate_hat)
	@test isapprox(ate_hat, -200, atol=10) == true
end