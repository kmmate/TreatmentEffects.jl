#=
Tests for matching.jl


Remarks
-------

=#


# testing ate_matchingestimator
@testset "ate_matchingestimator" begin
	# ============== Testing with real data set
	
	# read example dataset
	# subsampling
	n_subsample = 25_000
	subsample_idx = rand(1:60_000, n_subsample)
	datapath = normpath(joinpath(dirname(pathof(TreatmentEffects)), "..", "test\\data\\matching"))
	y_raw = Array(CSV.read(joinpath(datapath, "y.csv"), header=false)[:1])[subsample_idx]
	d_raw = Array(CSV.read(joinpath(datapath, "d.csv"), header=false)[:1])[subsample_idx]
	x_raw = Array(convert(Matrix, CSV.read(joinpath(datapath, "X.csv"),
		header=false, delim=';')))[subsample_idx, : ]
	y, d, x = drop_missing(y_raw, d_raw, x_raw)
	# set up model
	mam = MatchingModel(y, d, x)
	# estimate ATE
	#	---	test mathcing on covariates
	numberof_neighbours = 2
	matching_method = :covariates
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	println("ATE_hat $(matching_method) matching =  ", ate_hat)
	@test isapprox(ate_hat, -200, atol=50) == true
	#	---	test mathcing on covariates
	numberof_neighbours = 3
	matching_method = :propscore_logit
	ate_hat = ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	@test isapprox(ate_hat, -200, atol=50) == true
	
	# ============== Testing with Monte Carlo

	# DGP parameters
	n = 2000
	mu_0 = 3.
	mu_1 = 5.  # will imply ATE = 2.
	sigma_x1 = 0.9
	sigma_x2 = 0.7
	rho = 0.
	covar = rho * sigma_x1 * sigma_x2
	cov_matrix = [[sigma_x1 ^ 2 covar]; [covar sigma_x2 ^ 2]]
	distribution = MvNormal(cov_matrix)
	gen_x(n::Int64) = rand(distribution, n)
	gen_y0(x1::Array{Float64}, x2::Array{Float64}) = mu_0 .+ 0.8 * x1 .+ 1.5 * x2 .+
		0.9 * randn(length(x1))
	gen_y1(x1::Array{Float64}, x2::Array{Float64}) = mu_1 .+ 3.1 * x1 .+ 1.8 * x2 .+
		1.1 * randn(length(x1))
	gen_d(x1::Array{Float64}, x2::Array{Float64}) =  Int.(1.3 * x1 .+ 5 * x2 .>= randn(length(x1)))

	# monte carlo
	mc_reps = 1000
	atehat_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		x = gen_x(n)
		d = gen_d(x[1, :], x[2, :])
		y1 = gen_y1(x[1, :], x[2, :])
		y0 = gen_y0(x[1, :], x[2, :])
		y = @. d * y1 + (1 - d) * y0
		# set up model
		mam = MatchingModel(y, d, Array(x'))
		#	---	test mathcing on covariates
		numberof_neighbours = 1
		matching_method = :covariates
		ate_matchingestimator(mam, k=numberof_neighbours, matching_method=matching_method)
	end
	@test isapprox(atehat_sum / mc_reps, mu_1 - mu_0, atol=1.) == true
end



# testing att_matchingestimator
@testset "att_matchingestimator" begin
	# ============== Testing with real data set
	
	# read example dataset
	# subsampling
	n_subsample = 25_000
	subsample_idx = rand(1:60_000, n_subsample)
	datapath = normpath(joinpath(dirname(pathof(TreatmentEffects)), "..", "test\\data\\matching"))
	y_raw = Array(CSV.read(joinpath(datapath, "y.csv"), header=false)[:1])[subsample_idx]
	d_raw = Array(CSV.read(joinpath(datapath, "d.csv"), header=false)[:1])[subsample_idx]
	x_raw = Array(convert(Matrix, CSV.read(joinpath(datapath, "X.csv"),
		header=false, delim=';')))[subsample_idx, : ]
	y, d, x = drop_missing(y_raw, d_raw, x_raw)
	# set up model
	mam = MatchingModel(y, d, x)
	# estimate ATE
	#	---	test mathcing on covariates
	numberof_neighbours = 2
	matching_method = :covariates
	att_hat = att_matchingestimator(mam, k=numberof_neighbours,
		matching_method=matching_method)
	@test isapprox(att_hat, -200, atol=50) == true
	#	---	test mathcing on covariates
	numberof_neighbours = 2
	matching_method = :propscore_logit
	att_hat = ate_matchingestimator(mam, k=numberof_neighbours,
		matching_method=matching_method)
	@test isapprox(att_hat, -200, atol=50) == true

	# ============== Testing with Monte Carlo

	# DGP parameters
	n = 1000
	alpha_0 = 3.
	alpha_1 = 0.8
	alpha_2 = 1.5
	beta_0 = 5.
	beta_1 = 3.1
	beta_2 = 1.8 # implies ATT=beta0-alpha0+sum_{j=1,2}(alpha_j-beta_j)E[x_j|D=1]
	sigma_x1 = 0.9
	sigma_x2 = 1.1
	rho = 0.
	covar = rho * sigma_x1 * sigma_x2
	cov_matrix = [[sigma_x1 ^ 2 covar]; [covar sigma_x2 ^ 2]]
	distribution = MvNormal(cov_matrix)
	gen_x(n::Int64) = rand(distribution, n)
	gen_y0(x1::Array{Float64}, x2::Array{Float64}) = alpha_0 .+ alpha_1 * x1 .+
		alpha_2 * x2 .+ 0.9 * randn(length(x1))
	gen_y1(x1::Array{Float64}, x2::Array{Float64}) = beta_0 .+ beta_1 * x1 .+
		beta_2 * x2 .+ 1.1 * randn(length(x1))
	gen_d(x1::Array{Float64}, x2::Array{Float64}) =  Int.(1.3 * x1 .+
			5 * x2 .>= randn(length(x1)))

	# monte carlo
	mc_reps = 1000
	samples = @distributed (vcat) for rep in 1:mc_reps
		# generate data
		x = gen_x(n)
		d = gen_d(x[1, :], x[2, :])
		y1 = gen_y1(x[1, :], x[2, :])
		y0 = gen_y0(x[1, :], x[2, :])
		y = @. d * y1 + (1 - d) * y0
		# set up model
		mam = MatchingModel(y, d, Array(x'))
		#	---	test mathcing on covariates
		numberof_neighbours = 2
		matching_method = :propscore_logit
		att_hat = att_matchingestimator(mam, k=numberof_neighbours,
			matching_method=matching_method)
		[att_hat sum(x[1, d .== 1]) / sum(d) sum(x[2, d .== 1]) / sum(d)]
	end
	# unpack mc samples and average them out
	att_hat = sum(samples[:, 1]) / mc_reps
	mean_x1_d1 = sum(samples[:, 2]) / mc_reps
	mean_x2_d1 = sum(samples[:, 3]) / mc_reps
	att = beta_0 - alpha_0 + (beta_1 - alpha_1) * mean_x1_d1 +
		(beta_2 - alpha_2) * mean_x2_d1
	@test isapprox(att_hat, att, atol=1.5) == true
end


# testing ate_blockingestimator
@testset "ate_blockingestimator" begin
	# ============== Testing with real data set
	
	# read example dataset
	# subsampling
	n_subsample = 25_000
	subsample_idx = rand(1:60_000, n_subsample)
	datapath = normpath(joinpath(dirname(pathof(TreatmentEffects)), "..", "test\\data\\matching"))
	y_raw = Array(CSV.read(joinpath(datapath, "y.csv"), header=false)[:1])[subsample_idx]
	d_raw = Array(CSV.read(joinpath(datapath, "d.csv"), header=false)[:1])[subsample_idx]
	x_raw = Array(convert(Matrix, CSV.read(joinpath(datapath, "X.csv"),
		header=false, delim=';')))[subsample_idx, : ]
	y, d, x = drop_missing(y_raw, d_raw, x_raw)
	# set up model
	mam = MatchingModel(y, d, x)
	# estimate ATE
	ate_hat = ate_blockingestimator(mam)
	@test isapprox(ate_hat, -200, atol=50) == true
		
	# ============== Testing with Monte Carlo

	# DGP parameters
	n = 2000
	mu_0 = 3.
	mu_1 = 5.  # will imply ATE = 2.
	sigma_x1 = 0.9
	sigma_x2 = 0.7
	rho = 0.
	covar = rho * sigma_x1 * sigma_x2
	cov_matrix = [[sigma_x1 ^ 2 covar]; [covar sigma_x2 ^ 2]]
	distribution = MvNormal(cov_matrix)
	gen_x(n::Int64) = rand(distribution, n)
	gen_y0(x1::Array{Float64}, x2::Array{Float64}) = mu_0 .+ 0.8 * x1 .+ 1.5 * x2 .+
		0.9 * randn(length(x1))
	gen_y1(x1::Array{Float64}, x2::Array{Float64}) = mu_1 .+ 3.1 * x1 .+ 1.8 * x2 .+
		1.1 * randn(length(x1))
	gen_d(x1::Array{Float64}, x2::Array{Float64}) =  Int.(1.3 * x1 .+ 5 * x2 .>= randn(length(x1)))

	# monte carlo
	mc_reps = 1000
	atehat_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		x = gen_x(n)
		d = gen_d(x[1, :], x[2, :])
		y1 = gen_y1(x[1, :], x[2, :])
		y0 = gen_y0(x[1, :], x[2, :])
		y = @. d * y1 + (1 - d) * y0
		# set up model
		mam = MatchingModel(y, d, Array(x'))
		#	---	test mathcing on covariates
		block_boundaries = collect(0:0.1:1)
		ate_blockingestimator(mam, block_boundaries=block_boundaries)
	end
	@test isapprox(atehat_sum / mc_reps, mu_1 - mu_0, atol=1.) == true
end