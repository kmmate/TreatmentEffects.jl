#=

@author: Mate Kormos
@date: 12-Feb-2019

Define functions for nonparametric regression.

TODO:
	- profile the code
	- expand polynomial_features so that poldegree can be larger than no. of variables
	- bounded support kernels (triangular_kernel, uniform_kernel) often lead
	  to zero kernel weights and hence singular matrix when computing mhat with WLS
=#
using IterTools
#include("kernels.jl")
"""
    function localpoly_regression(x0::T where T<:Union{<:Real, Array{<:Real, 1}},
							  	  y::Array{<:Real, 1},
							      x::Array{<:Real},
							      bandwidth::Float64;
							      poldegree::Int64 = 2,
							      kernel::Function = triangular_kernel)

Local polynomial regression estimator.
Estimates the conditional expected value ``m(x)=E[y|X=x]``.

##### Arguments
- `x0`::T where T<:Union{<:Real, Array{<:Real, 1}} : m(x=x0) is estimated
- `y`::Array{<:Real, 1} : ``n``-large sample of outcome variable
- `x`::Array{<:Real} :  Sample of covariates. Either ``n``-long Array{<:Real,1}
						or ``n``-by-``d`` Array{<:Real, 2}
- `bandwidth`::Float64 : Bandwidth of kernel
- `poldegree`::Int64 : Degree of polynomial
- `kernel`::Function : One of the kernels in `kernels.jl`

##### Returns
- `mhat`::Float64 Estimated conditional expected value

##### Examples
```julia
n = 1000
x = [1.5 .* randn(n) 2. .* randn(n)]
y = 2 .* cos.(x[:, 1]) .+ (1 .- sin.(x[:, 2])) .^ 2 .* x[:, 2] .^ 3 + randn(n)
x0 = [0., 0.]
poldegree = 2
d = size(x)[2]
bandwidth = n ^ (- 1 / (2 * poldegree + d + 2))  # AMISE minimiser bandwidth choice
mhat = localpoly_regression(x0, y, x, bandwidth, poldegree=poldegree, kernel=uniform_kernel)
println("m(x0) = E[y| x0] = 2.", ". Estimated m(x0) = ", mhat)
```
"""
function localpoly_regression(x0::T where T<:Union{<:Real, Array{<:Real, 1}},
							  y::Array{<:Real, 1},
							  x::Array{<:Real},
							  bandwidth::Float64;
							  poldegree::Int64 = 2,
							  kernel::Function = triangular_kernel)
	# sizes and dimension checks
	n = length(y)
	dim = ndims(x)
	if dim == 1
		d = 1
	elseif dim == 2
		d = size(x)[2]  # number of regressors
	else
		error("`x` must be either 1 or 2 dimensional")
	end
	if poldegree > d
		error("`poldegree` cannot be larger than number of variables in `x`")	
	end
	if poldegree < 1
		error("`poldegree` cannot be smaller than 1")
	end
	# data checks
	if size(x)[1] != size(y)[1]
		error("number of observations in `x` and `y` must be the same")
	end
	# evaluation point checks
	if (dim == 1) && (ndims(x0) != 0)  # Array{T, 1} x
		error("the evaluation point `x0` must have the same number of variables as `x`")
	end
	if (dim == 2) && (ndims(x0) != 1)  # Array{T, 2} x
		error("the evaluation point `x0` must have the same number of variables as `x`")
	end

	# centered design matrix with polynomials and constant
	if d == 1
		xcenter = x .- x0
	else
		xcenter = x - repeat(x0', n)
	end
	designmatrix = polynomial_features(xcenter, poldegree)
	# compute weights with kernel
	kernelorder = 2
	u = xcenter / bandwidth  # input to kernel
	if d == 1
		w = [kernel(i, kernelorder=kernelorder) ^ 0.5 for i in u]
	else
		u_transpose = Array(u')  # for faster iteration k-by-n Array
		product_kernel(v::Array{<:Real}) = prod([kernel(i, kernelorder=kernelorder) for i in v])
		w = [product_kernel(u_transpose[:, col]) ^ 0.5 for col in 1:n]
	end

	# transform variables, compute mhat with WLS
	designmatrix_w = designmatrix .* w
	y_w = y .* w
	betahat = (designmatrix_w' * designmatrix_w) \ (designmatrix_w' * y_w)
	mhat = betahat[1]
	return mhat
end



#=
	Helper functions
=#

"""
    polynomial_features(x::Array{<:Real}, poldegree::Int64)

Creates polynomial features from the data in `x`, including intercept and interactions

Based on [ScikitLearn.Preprocessing](https://github.com/cstjean/ScikitLearn.jl/blob/master/src/preprocessing.jl).

##### Arguments
-`x`:Array{<:Real} : Data array. Either ``n``-long Array{<:Real, 1} or ``n``-by-``d``
	Array{<:Real, 2} where ``d`` is the number of variables,
	``n`` is the number of observations
- `poldegree`::Int64 : Highest degree of the polynomial to be created.
	Must be smaller or equal to the number of variables in `x`.
	For eg. `poldegree`=2 will include seconds order interactions and squres, 
	the main effects and an intercept.

##### Returns
- `x_out`::Array{<:Real, 2} : Data array with the added polynomial feaures
"""
function polynomial_features(x::Array{<:Real}, poldegree::Int64)
	# sizes, checks
	n = size(x)[1]
	dim = ndims(x)
	if dim == 1
		d = 1
	elseif dim == 2
		d = size(x)[2]
	else
		error("`x` must be 1 or 2 dimensional")
	end
	if poldegree > d
		error("`poldegree` cannot be larger than number of variables in `x`")	
	end
	if poldegree < 1
		error("`poldegree` cannot be smaller than 1")
	end
	
	# add features degree by degree
	no_outputvariables = sum([binomial(d, k) for k in 1:poldegree]) +
		d * (poldegree - 1) + 1
	x_out = zeros(n, no_outputvariables)
	x_out[:, 1] = ones(n)  # intercept
	x_out[:, 2:d+1] = x  # main effects x ^ 0
	if poldegree == 1
		return x_out
	end
	idx = d + 2
	for degree in 2:poldegree
		# add main effects, ie x^2, x^3 etc., variable by variable
		for k = 1:d
			x_out[:, idx] = x[:, k] .^ degree
			idx += 1
		end
		# add interaction effects, ie x1*x2 etc.
		ss = IterTools.subsets(collect(1:d), degree)
		for s in ss
			x_out[:, idx] = prod(x[:, s], dims=2)[:]  # multipy columns elementwise
			idx +=1
		end
	end
	return x_out
end