#=

@author: Mate Kormos
@date: 01-Feb-2019

Define auxiliary functions, estimators for internal use with model specific estimators.

=#

using GLM, DataFrames

"""
    predict_propscore(d::Array{<:Real, 1}, x::Array{<:Real}, estimation_method::Symbol)

Predict propensity score based on estimation as specified by `estimation_method`.

###### Arguments
- `d`::Array{<:Real, 1} : Binary outcome variable
- `x`::Array{<:Real} : Covariates
- `estimation_method`::Symbol : :logit or :nonparametric
"""
function predict_propscore(d::Array{<:Real, 1}, x::Array{<:Real}, estimation_method::Symbol)
	if in(estimation_method, [:logit, :nonparametric]) == false
		error("`estimation_method` must be either :logistic or :nonparametric")
	# logit
	elseif estimation_method == :logit
		df = DataFrame(x=x, d=d)
		pscore_hat = logit_predict(df)
		return pscore_hat
	# nonparametric
	elseif estimation_method == :nonparametric
		error("nonparametric propensity score estimation is not implemented")
		pscore_hat = nonparametric_predict(d, x)
		return pscore_hat
	end
end

#=
Parametric auxiliary estimators for internal use
=#

"""
    logit_predict(df::DataFrame)  

Predict the propensity score P(D=1|X) with logistic regression.
"""
function logit_predict(df::DataFrame)
	logit = glm(@formula(d ~ x), df, Bernoulli(), LogitLink())
    pscore_hat = predict(logit)
    return pscore_hat
end


#=
Nonparametric auxiliary estimators for internal use
=#

"""
Predict the propensity score P(D=1|X) with nonparametric regression.
"""
function nonparametric_predict()

end