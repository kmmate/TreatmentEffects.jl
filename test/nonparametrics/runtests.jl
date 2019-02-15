#=
Test for nonparametrics

@author: Mate Kormos
@date: 18-Jan-2019

include("C:\\Users\\Máté\\Dropbox\\tinbergen\\TreatmentEffects.jl\\test\\nonparametrics\\runtests.jl")
=#
using Test,
	  Random,
	  Distributions,
	  Distributed

include("..\\..\\src\\nonparametrics\\kernels.jl")
include("..\\..\\src\\nonparametrics\\np_regression.jl")

tests = ["test_np_regression.jl"]


Random.seed!(19970531)

println("Running tests...")
for t in tests
	println("testing: ", t[6:end])
	include(t)
	println("\t\033[1m\033[32mPASSED\033[0m: $(t)")
end