#=
Tests for matching.jl


Remarks
-------



=#

# testing ate_matchingestimator
@testset "ate_matchingestimator" begin
	# read example dataset
	datapath = joinpath(dirname(pathof(TreatmentEffects)), "\\test\\data\\mathcing\\")
	y = CSV.read(joinpath(datapath, "y.csv"), header=false)
	d = CSV.read(joinpath(datapath, "d.csv"), header=false)
	x = CSV.read(joinpath(datapath, "X.csv"), header=false)
	# set up model
	mam = MatchingModel(y, d, x)
	# estimate ATE
	numberof_neighbours = 1
	matching_method = :covariates
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	println("ATE_hat = ", ate_hat)
	@test isapprox(ate_hat, -200, atol=10) == true
end