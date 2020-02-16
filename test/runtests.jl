#=
Test for TreatmentEffects.jl

@author: Mate Kormos
@date: 18-Jan-2019


=#
using TreatmentEffects,
	  Test,
	  Random,
	  Distributions,
	  Distributed,
	  SharedArrays,
	  CSV,
	  DataFrames

tests = [#"test_exogenous_participation.jl",
		 #"test_data_cleaning.jl",
		 #"test_matching.jl",
		 "test_rdd.jl",
		 "test_exogenous_noncompliance.jl",
		 "test_endogenous_noncompliance.jl",
		 "test_perfect_compliance.jl"]


Random.seed!(19970531)

println("Running tests...")
for t in tests
	println("testing: ", t[6:end])
	include(t)
	println("\t\033[1m\033[32mPASSED\033[0m: $(t)")
end