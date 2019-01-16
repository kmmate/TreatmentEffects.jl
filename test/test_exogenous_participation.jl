#=
Test for exogenous_participation.jl


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
# Then Y = mu_0 + (mu_1 - mu_0)D + (e_1-e_0)D.
# [e_1, e_0] is sampled from multivariate normal distribution,
  such that they are correlated.
# D is sampled from Bernoulli
=#
@testset "ate_estimator" begin
	# DGP parameters
	n = 1000
	mu_0 = 3
	mu_1 = 5  # implies that ATE = 2
	sigma_e0 = 2.3
	sigma_e1 = 4.1
	rho = 0.7
	covar = rho * sigma_e0 * sigma_e1
	cov_matrix = [[sigma_e0 ^ 2 covar]; [covar sigma_e1 ^ 2]]
	distribution = MvNormal(cov_matrix)

	# monte carlo
	mc_reps = 10
	atehat_sum = 0.
	for rep in 1:mc_reps
		# generate data
		d = rand([0, 1], n)
		e = rand(distribution, n)
		e_0 = e[1, :]
		e_1 = e[2, :]
		y = mu_0 .+ (mu_1 - mu_0) .* d .+ (e_1 .- e_0) .* d
		# set up model
		epm = ExogenousParticipationModel(y, d)
		println(sum(epm.y[epm.d == 1]) / sum(epm.d))
		atehat_sum += ate_estimator(epm)  # cumulative sum so that it can be averaged out
	end
	println(atehat_sum)
	@test isapprox(atehat_sum / reps, mu_1 - mu_0, atol=1) == true
end