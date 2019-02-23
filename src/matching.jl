#=

@author: Mate Kormos
@date: 31-Jan-2019

Defines methods (estimators) for MatchingModel <: CIAModel type.

TODO:
	- bias correction for matching estimator
=#

using StatsBase, Distances

"""
    ate_matchingestimator(m::MatchingModel;
						  k::Int64 = 1,
						  matching_method::Symbol = :propscore_logit,
						  np_options::Dict = Dict(:kernel => gaussian_kernel,
							   						   :bandwidth => :optimal,
							   						   :poldegree => 2)
							   )


Estimate Average Treatment Effect (ATE) with k-nearest neighbour matching.

##### Arguments
- `m`::MatchingModel : MatchingModel model type
- `k`::Int64 : Number of nearest neighbours
- `matching_method`::Symbol : Method used for matching,
	one of :propscore_logit, :propscore_nonparametric, :covariates
- `np_options`::Dict : Nonparametric propensity score estimation options,
	Dict(:kernel => triangular_kernel, :bandwidth => :optimal, :poldegree => 2),
	where :kernel is function in `kernels.jl`, :bandwidth is either :optimal
	for AMISE optimal bandwidth or a Float64,  :poldegree is Int64, 
	the polynomial degree in the local polynomial regression

!!! warning

In nonparametric propensity score estimation, bounded support kernels 
(`triangular_kernel`, `uniform_kernel`) may lead to zero kernel weights
and hence singular matrix when estimating propensity score with WLS.
Use `gaussian_kernel` or `epanechnikov_kernel` instead.

##### Returns
- `ate_hat`::Float64 : Estimated ATE

##### Examples
```julia
using TreatmentEffects, CSV
y = Array(CSV.read("y_data.csv")[:1])  # [:1] to get Array{T, 1}
d = Array(CSV.read("d_data.csv")[:1])
x = Array(convert(Matrix, CSV.read("x_data.csv")))
mam = MatchingModel(y, d, x)
ate_matchingestimator(mam, k=2, matching_method=:covariates)
```
"""
function ate_matchingestimator(m::MatchingModel;
							   k::Int64 = 1,
							   matching_method::Symbol = :propscore_logit,
							   np_options::Dict = Dict(:kernel => gaussian_kernel,
							   						   :bandwidth => :optimal,
							   						   :poldegree => 2)
							   )
	# separate treatment and control group
	n = size(m.y)[1]  # sample size
    n_t = Int(sum(m.d))  # no. of treated units
    n_c = Int(n - n_t)  # no. of control units
    y_t = m.y[m.d .== 1]  # y of those in treatment group
    y_c = m.y[m.d .== 0]  # y of those in control group
    x_t = m.x[m.d .== 1, :]  # x of those in treatment group
    x_c = m.x[m.d .== 0, :]  # x of those in control group
    
    # (predict porpensity score) and compute distances based on matching_method
    distance_matrix = zeros(n_t, n_c)
    if matching_method == :covariates
    	# estimate sample covariance matrix for distance computation
    	cov_hat = StatsBase.cov(m.x)
    	# n_t-by-n_c array with  element (i, j) = distance between 'i'th treated and 'j'th control units
    	Distances.pairwise!(distance_matrix, Mahalanobis(inv(cov_hat)), x_t', x_c')
    elseif matching_method == :propscore_logit || matching_method == :propscore_nonparametric
    	# estimate propensity score
    	if matching_method == :propscore_logit
    		phat = predict_propscore(m.d, m.x, :logit)
    	elseif matching_method == :propscore_nonparametric
    		phat = predict_propscore(m.d, m.x, :nonparametric, np_options=np_options)
    	else
    		error("`propscore_estimation` mumst be either :logit or :nonparametric")
    	end
    	phat_t = phat[m.d .== 1]  # propensity score in treatment group
    	phat_c = phat[m.d .== 0]  # propensity score in control group
    	# n_t-by-n_c array with  element (i, j) = distance between 'i'th treated and 'j'th control units
    	Distances.pairwise!(distance_matrix, Euclidean(), phat_t[: ,: ]', phat_c[:, :]')
    end
    
    # estimate a control outcome, Y(0), for each treated unit with matching
    yc_hat = zeros(n_t)
    for treated_unit in 1:n_t
        @inbounds yc_hat[treated_unit] = estimate_counterfactual(k,
        	distance_matrix[treated_unit, :], y_c)
    end
    # estimate a treated outcome, Y(1), for each control unit with matching
    yt_hat = zeros(n_c)
    for control_unit in 1:n_c
        @inbounds yt_hat[control_unit] = estimate_counterfactual(k,
        	distance_matrix[:, control_unit], y_t)
    end
    # compute the ATE estimator: 1/n * sum_i(Yhat_i(1)-Yhat_i(0))
    ate_hat = 1 / n * (sum(y_t - yc_hat) + sum(yt_hat - y_c)) 
    return ate_hat
end


"""
    att_matchingestimator(m::MatchingModel;
						  k::Int64 = 1,
						  matching_method::Symbol = :propscore_logit,
						  np_options::Dict = Dict(:kernel => gaussian_kernel,
						  						  :bandwidth => :optimal,
						 						  :poldegree => 2)
							   )

Estimate Average Treatment Effect (ATT) with k-nearest neighbour matching.

##### Arguments
- `m`::MatchingModel : MatchingModel model type
- `matching_method`::Symbol : Method used for matching,
	one of :propscore_logit, :propscore_nonparametric, :covariates
- `k`::Int64 : Number of nearest neighbours
- `np_options`::Dict : Nonparametric propensity score estimation options,
	Dict(:kernel => triangular_kernel, :bandwidth => :optimal, :poldegree => 2),
	where :kernel is function in `kernels.jl`, :bandwidth is either :optimal
	for AMISE optimal bandwidth or a Float64,  :poldegree is Int64, 
	the polynomial degree in the local polynomial regression

!!! warning

In nonparametric propensity score estimation, bounded support kernels 
(`triangular_kernel`, `uniform_kernel`) may lead to zero kernel weights
and hence singular matrix when estimating propensity score with WLS.
Use `gaussian_kernel` or `epanechnikov_kernel` instead.

##### Returns
- `att_hat`::Float64 : Estimated ATE

##### Examples
```julia
using TreatmentEffects, CSV
y = Array(CSV.read("y_data.csv")[:1])  # [:1] to get Array{T, 1}
d = Array(CSV.read("d_data.csv")[:1])
x = Array(convert(Matrix, CSV.read("x_data.csv")))
mam = MatchingModel(y, d, x)
att_matchingestimator(mam, k=2, matching_method=:covariates)
```
"""
function att_matchingestimator(m::MatchingModel;
							   k::Int64 = 1,
							   matching_method::Symbol = :propscore_logit,
							   np_options::Dict = Dict(:kernel => gaussian_kernel,
							   						   :bandwidth => :optimal,
							   						   :poldegree => 2)
							   )
	# separate treatment and control group
	n = size(m.y)[1]  # sample size
    n_t = Int(sum(m.d))  # no. of treated units
    n_c = Int(n - n_t)  # no. of control units
    y_t = m.y[m.d .== 1]  # y of those in treatment group
    y_c = m.y[m.d .== 0]  # y of those in control group
    x_t = m.x[m.d .== 1, :]  # x of those in treatment group
    x_c = m.x[m.d .== 0, :]  # x of those in control group
    
    # (predict porpensity score) and compute distances based on matching_method
    distance_matrix = zeros(n_t, n_c)
    if matching_method == :covariates
    	# estimate sample covariance matrix for distance computation
    	cov_hat = StatsBase.cov(m.x)
    	# n_t-by-n_c array with  element (i, j) = distance between 'i'th treated and 'j'th control units
    	Distances.pairwise!(distance_matrix, Mahalanobis(inv(cov_hat)), x_t', x_c')
    elseif matching_method == :propscore_logit || matching_method == :propscore_nonparametric
    	# estimate propensity score
    	if matching_method == :propscore_logit
    		phat = predict_propscore(m.d, m.x, :logit)
    	elseif matching_method == :propscore_nonparametric
    		phat = predict_propscore(m.d, m.x, :nonparametric, np_options=np_options)
    	else
    		error("`propscore_estimation` mumst be either :logit or :nonparametric")
    	end
    	phat_t = phat[m.d .== 1]  # propensity score in treatment group
    	phat_c = phat[m.d .== 0]  # propensity score in control group
    	# n_t-by-n_c array with  element (i, j) = distance between 'i'th treated and 'j'th control units
    	Distances.pairwise!(distance_matrix, Euclidean(), phat_t[: ,: ]', phat_c[:, :]')
    end
    
    # estimate a control outcome, Y(0), for each treated unit with matching
    yc_hat = zeros(n_t)
    for treated_unit in 1:n_t
        @inbounds yc_hat[treated_unit] = estimate_counterfactual(k,
        	distance_matrix[treated_unit, :], y_c)
    end
    # compute the ATT estimator: 1/n_t * sum_{i in treated}(Yhat_i(1)-Yhat_i(0))
    att_hat = 1 / n_t * sum(y_t - yc_hat) 
    return att_hat
end


"""
    ate_blockingestimator(m::MatchingModel;
					      propscore_estimation::Symbol=:logit,
						  block_boundaries::Array{Float64, 1}=[0., 0.2, 0.4, 0.6, 0.8, 1.],
						  np_options::Dict = Dict(:kernel => gaussian_kernel,
						   					      :bandwidth => :optimal,
							   					  :poldegree => 2))

Estimate Average Treatment Effect (ATE) with blocking on propensity score.

Note: blocking estimation as also known as stratification.

!!! warning

The estimator is quite sensitive to overlap assumption:
it requires that the propensity score in the treated group overlaps with the
propensity score in the control group. Formally:
support(Prob(treatment|x)|treated group)=support(Prob(treatment|x)|control group).
If the assumption is violated, it may return `NaN`.


##### Arguments
- `m`::MatchingModel : MatchingModel model type
- `propscore_estimation`::Symbol : Estimator used to estimate the propensity score.
	Either :logit or :nonparametric.
- `block_boundaries`::Array{Float64, 1} : Boundaries of estimated propensity score
	used to create strata/blocks. `length(block_boundaries)` = number of blocks + 1.
	Elements must be in increasing order, with each element between 0 and 1.
	Must incluide 0 and 1 as endpoints.
- `np_options`::Dict : Nonparametric propensity score estimation options,
	Dict(:kernel => triangular_kernel, :bandwidth => :optimal, :poldegree => 2),
	where :kernel is function in `kernels.jl`, :bandwidth is either :optimal
	for AMISE optimal bandwidth or a Float64,  :poldegree is Int64, 
	the polynomial degree in the local polynomial regression


!!! warning

In nonparametric propensity score estimation, bounded support kernels 
(`triangular_kernel`, `uniform_kernel`) often lead to zero kernel weights
and hence singular matrix when estimating propensity score with WLS.
Use `gaussian_kernel` or `epanechnikov_kernel` instead.

##### Returns
- `ate_hat`::Float64 : Estimated ATE

##### Examples
```julia
using TreatmentEffects, CSV
y = Array(CSV.read("y_data.csv")[:1])  # [:1] to get Array{T, 1}
d = Array(CSV.read("d_data.csv")[:1])
x = Array(convert(Matrix, CSV.read("x_data.csv")))
mam = MatchingModel(y, d, x)
ate_blockingestimator(mam)
```
"""
function ate_blockingestimator(m::MatchingModel;
							   propscore_estimation::Symbol=:logit,
							   block_boundaries::Array{Float64, 1}=[0., 0.2, 0.4, 0.6, 0.8, 1.],
							   np_options::Dict = Dict(:kernel => gaussian_kernel,
							   						   :bandwidth => :optimal,
							   						   :poldegree => 2))
	# checks
	if block_boundaries[1] != 0
		error("`block_boundaries` must contain 0 as its first element")
	elseif block_boundaries[end] != 1
		error("`block_boundaries` must contain 1 as its last element")
	end
	if sort(block_boundaries) != block_boundaries
		error("`block_boundaries` must be strictly increasing")
	end
	# sizes
	n = size(m.y)[1]  # sample size
	n_blocks = length(block_boundaries) - 1
	# propensity score estimation
	if propscore_estimation == :logit
		phat = predict_propscore(m.d, m.x, :logit)
	elseif propscore_estimation == :nonparametric
		phat = predict_propscore(m.d, m.x, :nonparametric, np_options=np_options)
	else
		error("`propscore_estimation` mumst be either :logit or :nonparametric")
	end
	# censor extreme values
	phat[phat .< 0] = 0. .* ones(sum(phat .< 0))
	phat[phat .> 1] = 1. .* ones(sum(phat .> 1))
	# divide the observations into blocks based on propensity score estimates
	block_labeller(p::Float64) = [i for i in 1:n_blocks if
		block_boundaries[i] <= p <= block_boundaries[i+1]][1]
	block_labels = map(block_labeller, phat)
    atehat_blocks = zeros(n_blocks)
    samplesize_blocks = zeros(n_blocks)
    # estimate ATE in each block
    for block_label in 1:n_blocks
    	# treatment and control outcomes in the block
    	yt_block = m.y[(m.d .== 1) .* (block_labels .== block_label)]
    	yc_block = m.y[(m.d .== 0) .* (block_labels .== block_label)]
    	# sample sizes in the block
    	nt_block = length(yt_block)
    	nc_block = length(yc_block)
    	n_block = nt_block + nc_block
    	atehat_blocks[block_label] = sum(yt_block) / nt_block - sum(yc_block) / nc_block
    	samplesize_blocks[block_label] = n_block
    end
    # estimate ATE from block estimates
    ate_hat = (atehat_blocks' * samplesize_blocks) / n
    return ate_hat
end

#=
	Auxiliary functions for internal use
=#


"""
    estimate_counterfactual(k::Int64, distance_array::Array{Float64, 1}, y_pool::Array{<:Real, 1})

Estimate the counterfactual potential outcome (Y(1) or Y(0)) with matching.

The outcome is estimated as an average of the outcomes of
the k-nearest units in the other (treatment or control) group.

##### Arguments
- `k`::Int64 : Number of nearest neighbours used for matching
- `distance_array`::Array{Float64, 1} : Distances between the unit to match to
	and the units in the other group
- `y_pool`::Array{<:Real, 1} : Observed outcomes of the units in the other group

##### Returns
- `y_counterfactual_hat`::Float64 : Estimated counterfactual outcome
"""
function estimate_counterfactual(k::Int64, distance_array::Array{Float64, 1}, y_pool::Array{<:Real, 1})
    # indices of units in the other group:
    # the first index is that of the other-group-unit with the smallest distance
    # from the unit to match to
    ordered_index = sortperm(distance_array)
    # average the outcomes of the k nearest other-group-units to get Yhat_i(D)
    y_counterfactual_hat = sum(y_pool[ordered_index[1:k]]) / k
    return y_counterfactual_hat
end