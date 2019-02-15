#=

@author: Mate Kormos
@date: 12-Feb-2019

Testing np_regression.jl

=#

# testing localpoly_regression
@testset "localpoly_regression" begin
	# DGP parameters
	function gen_pure_y(x1::T where T<:Union{<:Real, Array{<:Real, 1}},
				   		x2::T where T<:Union{<:Real, Array{<:Real, 1}}) 
		return 2 .* cos.(x1) .+ (1 .- sin.(x2)) .^ 2 .* x2 .^ 3
	end
	n = 1000
	x0 = [1., 0.]  # point of evaluation

	# monte carlo
	mc_reps = 1000
	poldegree = 2
	# test with different kernels
	for kernel in [uniform_kernel, triangular_kernel, gaussian_kernel, epanechnikov_kernel]
		y0_hat = @distributed (+) for rep in 1:mc_reps
			x = [1.5 .* randn(n) 2. .* randn(n)]
			y = gen_pure_y(x[:, 1], x[:, 2]) + 1.5 * randn(n)  # add noise
			d = size(x)[2]
			bandwidth = n ^ (- 1 / (2 * poldegree + d + 2))  # AMISE minimiser bandwidth choice
			yhat = localpoly_regression(x0, y, x, bandwidth, poldegree=poldegree,
				kernel=kernel)
			end
		println("kernel: $(kernel)", ", m_hat = ", y0_hat / mc_reps, ", m = ", gen_pure_y(x0[1], x0[2]))
		@test isapprox(y0_hat / mc_reps, gen_pure_y(x0[1], x0[2]), atol=0.3) == true
	end
end