#=
Tests for exogenous_participation.jl


Remarks
-------

In the ExogenousParticipation model potential outcomes are independent of
participation,  that is ``[Y(0), Y(1)]`` is independent of ``D``.

It is assumed that the only source of uncertainty is sampling. 
Ie, the observed dataset {(y_i, d_i)} i=1,...,n is and i.i.d. sample from a population,
and the experimenter has no control over treatment participation, ``D``.

To test the model methods the following data generating process (DGP) is used:

# Observed outcome: Y = Y(0) + (Y(1)-Y(0))D by definition.
# Y(k) potential outcome is decomposed as Y(k) = E[Y(k)|D] + e_k where E[e_k|D]=0
  (every random variable admits such a decomposition).
# By independence E[Y(k)|D]=E[Y(k)].
# Let mu_k = E[Y(k)], so ATE = E[Y(1)-Y(0)] = mu_1 - mu_0.
# Then Y = mu_0 + (mu_1 - mu_0)D + e_0 + (e_1-e_0)D.
# [e_1, e_0] is sampled from multivariate normal distribution,
  such that they are correlated.
# D is sampled from Bernoulli
=#


# testing ate_estimator
@testset "ate_estimator" begin
	# ============== Testing under Full Homoskedasticity: 
	#			E[e_{0i}^2|d_i]=E[e_{1i}^2|d_i]=sigma
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	sigma_e0 = 2.3
	sigma_e1 = sigma_e0
	rho = 0.7
	covar = rho * sigma_e0 * sigma_e1
	cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
	distribution = MvNormal(cov_matrix)

	# monte carlo
	mc_reps = 999
	atehat_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)
		e = rand(distribution, n)
		e_0 = e[1, :]
		e_1 = e[2, :]
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		ate_estimator(epm)  # cumulative sum so that it can be averaged out
	end
	@test isapprox(atehat_sum / mc_reps, mu_1 - mu_0, atol=0.1) == true

	
	# ============== Testing under Semi - Heteroskedasticity: 
	#			E[e_{0i}^2|d_i]=sigma_0^2 for all i, E[e_{1i}^2|d_i]=sigma_1^2 for all i
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	sigma_e0 = 2.3
	sigma_e1 = 4.3
	rho = 0.7
	covar = rho * sigma_e0 * sigma_e1
	cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
	distribution = MvNormal(cov_matrix)

	# monte carlo
	mc_reps = 999
	atehat_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)
		e = rand(distribution, n)
		e_0 = e[1, :]
		e_1 = e[2, :]
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		ate_estimator(epm)  # cumulative sum so that it can be averaged out
	end
	@test isapprox(atehat_sum / mc_reps, mu_1 - mu_0, atol=0.1) == true


	# ============== Testing under Heteroskasticity: 
	#			E[e_{0i}^2|d_i] = d_i * sigma_{01}^2 + (1 - d_i) * sigma_{00}^2 for all i,
	#			E[e_{1i}^2|d_i] = d_i * sigma_{11}^2 + (1 - d_i) * sigma_{10}^2 for all i
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	# standard deviations of e_0
	sigma_00 = 2.3  
	sigma_01 = 4.3
	# standard deviations of e_1
	sigma_10 = 3.4
	sigma_11 = 2.7
	rho = 0.7

	# monte carlo
	mc_reps = 999
	atehat_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)
		# heteroskastic error terms
		e_0 = Array{Float64, 1}(undef, n)
		e_1 = Array{Float64, 1}(undef, n)
		for i in 1:n
			sigma_e0 = sigma_01 * d[i] + sigma_00 * (1 - d[i])
			sigma_e1 = sigma_11 * d[i] + sigma_10 * (1 - d[i])
			covar = rho * sigma_e0 * sigma_e1
			cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
			distribution = MvNormal(cov_matrix)
			e = rand(distribution, 1)
			e_0[i] = e[1]
			e_1[i] = e[2]
		end
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		ate_estimator(epm)  # cumulative sum so that it can be averaged out
	end
	@test isapprox(atehat_sum / mc_reps, mu_1 - mu_0, atol=0.1) == true
end


# testing bootstrap_distribution
@testset "bootstrap_distribution" begin
	# ============== Testing under Full Homoskedasticity: 
	#			E[e_{0i}^2|d_i]=E[e_{1i}^2|d_i]=sigma
	# test the mean and asymptotic distribution
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	p = 0.5  # Prob(D=1)
	sigma_e0 = 2.3
	sigma_e1 = sigma_e0
	rho = 0.7
	covar = rho * sigma_e0 * sigma_e1
	cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
	distribution = MvNormal(cov_matrix)

	# monte carlo
	mc_reps = 999
	# to store cumulative sum of the mean of the bootstrap distribution
	bsdistribution_meanvar = @distributed (vcat) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)  # implies Prob(d=1)=0.5
		e = rand(distribution, n)
		e_0 = e[1, :]
		e_1 = e[2, :]
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		# obtain bootstrap distribution
		bsdistribution = bootstrap_distribution(epm)
		# compute bootstrap distribution mean
		bs_mean = mean(bsdistribution)
		# compute bootstrap distribution asymptotic variance
		bs_asymvar = var(sqrt(n) .* (bsdistribution .- bs_mean))
		[bs_mean bs_asymvar]
	end
	# unpack bootstrap means and asymptotic variances from MC reps, average them out
	bsdistribution_mean = mean(bsdistribution_meanvar[:, 1])
	bsdistribution_asymvar = mean(bsdistribution_meanvar[:, 2])
	# test bootstrap variance
	@test isapprox(bsdistribution_mean, mu_1 - mu_0, atol=0.1) == true
	# test bootstrap variance: sqrt(n)(betahat-E[beta]) ~ N(0, sigma^2 * Prob(D=1))
	@test isapprox(bsdistribution_asymvar, sigma_e0 ^ 2 / (p - p ^ 2), atol=0.1) == true

	# ============== Testing under Heteroskasticity: 
	#			E[e_{0i}^2|d_i] = d_i * sigma_{01}^2 + (1 - d_i) * sigma_{00}^2 for all i,
	#			E[e_{1i}^2|d_i] = d_i * sigma_{11}^2 + (1 - d_i) * sigma_{10}^2 for all i
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	# standard deviations of e_0
	sigma_00 = 2.3  
	sigma_01 = 4.3
	# standard deviations of e_1
	sigma_10 = 3.4
	sigma_11 = 2.7
	rho = 0.7

	# monte carlo
	mc_reps = 999
	bs_distributionmean_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)
		# heteroskastic error terms
		e_0 = SharedArray{Float64, 1}(n)
		e_1 = SharedArray{Float64, 1}(n)
		@distributed for i in 1:n
			sigma_e0 = sigma_01 * d[i] + sigma_00 * (1 - d[i])
			sigma_e1 = sigma_11 * d[i] + sigma_10 * (1 - d[i])
			covar = rho * sigma_e0 * sigma_e1
			cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
			distribution = MvNormal(cov_matrix)
			e = rand(distribution, 1)
			e_0[i] = e[1]
			e_1[i] = e[2]
		end
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		mean(bootstrap_distribution(epm, bs_reps = 100))
	end
	@test isapprox(bs_distributionmean_sum / mc_reps, mu_1 - mu_0, atol=0.1) == true
end



# testing bootstrap_htest
@testset "bootstrap_htest" begin
	# ============== Testing under Full Homoskedasticity: 
	#			E[e_{0i}^2|d_i]=E[e_{1i}^2|d_i]=sigma
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	sigma_e0 = 2.3
	sigma_e1 = sigma_e0
	rho = 0.7
	covar = rho * sigma_e0 * sigma_e1
	cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
	distribution = MvNormal(cov_matrix)
	# test H_0: ATE = 2. against H_1: ATE != 2
	ate_0 = 2.
	# monte carlo
	mc_reps = 999
	bs_pvalue_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)  # implies Prob(d=1)=0.5
		e = rand(distribution, n)
		e_0 = e[1, :]
		e_1 = e[2, :]
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		# test H_0: ATE = 2. against H_1: ATE != 2
		bs_pvalue = bootstrap_htest(epm, ate_0)
	end
	@test isapprox(bs_pvalue_sum / mc_reps, 1, atol=0.1) == true

	# ============== Testing under Heteroskasticity: 
	#			E[e_{0i}^2|d_i] = d_i * sigma_{01}^2 + (1 - d_i) * sigma_{00}^2 for all i,
	#			E[e_{1i}^2|d_i] = d_i * sigma_{11}^2 + (1 - d_i) * sigma_{10}^2 for all i
	
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	# standard deviations of e_0
	sigma_00 = 2.3  
	sigma_01 = 4.3
	# standard deviations of e_1
	sigma_10 = 3.4
	sigma_11 = 2.7
	rho = 0.7
	# test H_0: ATE = 2. against H_1: ATE != 2
	ate_0 = 2.
	# monte carlo
	mc_reps = 999
	bs_pvalue_sum = @distributed (+) for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)
		# heteroskastic error terms
		e_0 = SharedArray{Float64, 1}(n)
		e_1 = SharedArray{Float64, 1}(n)
		@distributed for i in 1:n
			sigma_e0 = sigma_01 * d[i] + sigma_00 * (1 - d[i])
			sigma_e1 = sigma_11 * d[i] + sigma_10 * (1 - d[i])
			covar = rho * sigma_e0 * sigma_e1
			cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
			distribution = MvNormal(cov_matrix)
			e = rand(distribution, 1)
			e_0[i] = e[1]
			e_1[i] = e[2]
		end
		y = @. mu_0 + (mu_1 - mu_0) * d + e_0 + (e_1 - e_0) * d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		# test H_0: ATE = 2. against H_1: ATE != 2
		bs_pvalue = bootstrap_htest(epm, ate_0)
	end
	@test isapprox(bs_pvalue_sum / mc_reps, 1, atol=0.1) == true
end