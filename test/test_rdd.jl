#=

Tests for rdd.jl

=#

# testing rdd_sharpestimator
@testset "rdd_sharpestimator" begin
	# DGP parameters
	n = 1000
	mu_0 = 3.
	mu_1 = 5.  # will imply tau = 2
	cutoff = 0.
	gen_x(n::Int64) = cutoff .+ 1.1 .* randn(n)
	gen_y0(x::Array{Float64, 1}) = mu_0 .+ 0.8 * x .* sin.(x) .+ 0.9 * randn(length(x))
	gen_y1(x::Array{Float64, 1}) = mu_1 .+ 1.6 * x .* sin.(x) .+ 1.1 * randn(length(x))
	gen_d(x::Array{Float64, 1}) =  Int.(x .>= cutoff)
	# estimation options
	poldegree = 1
	bandwidth = [1.5, n ^ (- 1 / (2 * poldegree + 3))]  # AMISE minimiser bandwidth choice
	np_options = Dict(:kernel => triangular_kernel, :poldegree => poldegree)
	lscv_options = Dict(:subsampling => true, :subsamplesize => round(Int, length(rdm.y) / 2),
                    	:window => :median)
	# monte carlo
	mc_reps = 500
	tauhat_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		x = gen_x(n)
		d = gen_d(x)
		y0 = gen_y0(x)
		y1 = gen_y1(x)
		y = @. d * y1 + (1 - d) * y0
		# model setup
		rdm = RDDModel(y, d, x, cutoff)
		# estimation
		(tau_hat, h_opt) = rdd_sharpestimator(rdm, bandwidth, bias_correction=false,
	    		  		   np_options=np_options, lscv_options=lscv_options)
		tau_hat
	end
	println("Predicted CATE($(cutoff)) = $(tauhat_sum / mc_reps). tau = $(tau).")
	@test isapprox(tauhat_sum / mc_reps, mu_1 - mu_0, atol=0.2) == true
end