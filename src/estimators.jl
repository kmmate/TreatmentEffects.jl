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

##### Returns
- `pscore_hat`::Array{Float64, 1} : Predicted propensity score
"""
function predict_propscore(d::Array{<:Real, 1}, x::Array{<:Real}, estimation_method::Symbol)
	if in(estimation_method, [:logit, :nonparametric]) == false
		error("`estimation_method` must be either :logistic or :nonparametric")
	# logit
	elseif estimation_method == :logit
		df_d = DataFrame(d[:, :], [:d])
		df_x_names = [Symbol("x$i") for i in 1:size(x)[2]]
		df_x = DataFrame(x, df_x_names)
		df = hcat(df_d, df_x)
		pscore_hat = logit_predict(df, df_x_names)
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
    logit_predict(df::DataFrame, df_x_names::Array{Symbol, 1})  

Predict the propensity score P(D=1|X) with logistic regression.

##### Arguments
- `df`::DataFrame : DataFrame containing ``d`` and ``x``
- `df_x_names`::Array{Symbol, 1} : Names of `x` variables

##### Returns
- `pscore_hat`::Array{Float64, 1} : Predicted propensity score
"""
function logit_predict(df::DataFrame, df_x_names::Array{Symbol, 1})
	# run `d` on all the covariates
	lhs = :d
	rhs = Expr(:call, :+, df_x_names...)
	logit = glm(@eval(@formula($(lhs) ~ $(rhs))), df, Bernoulli(), LogitLink())
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