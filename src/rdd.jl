#=

@author: Mate Kormos
@date: 02-Mar-2019

Defines methods (estimators) for RDDModel <: CIAModel type.

TODO:
=#
using StatsBase
"""
    function rdd_sharpestimator(m::RDDModel,
							assignment_rule::Symbol,
							bandwidth::Union{T where T<:Real, Array{<:Real, 1}},
							bias_correction::Bool = true,
							np_options::Dict = Dict(:kernel => triangular_kernel,
													:poldegree => 1),
							lscv_options::Dict = Dict(:subsamplesize => length(m.y) / 2,
													  :window => :median))

Estimate sharp regression discontinuity design estimand  ``E[Y(1)-Y(0)| X=cutoff]``
nonparametrically.

##### Arguments
- `m`::RDDModel : RDDModel type
- `assignment_rule`::Symbol : Either `above_cutoff` or `below_cutoff`, to indicate
	the assignment rules ``d_i = 1(x_i>=m.cutoff)`` or ``d_i = 1(x_i<=m.cutoff)``,
	respectively
- `bandwidth`::Union{T where T<:Real, Array{<:Real, 1}} : Bandwidth used for 
	nonparametric estimation. If a scalar T<:Real is given, than the given value is
	used for estimation. If an Array{<:Real, 1} is given, leave-one-out least squares
	cross-validation is performed to find the optimal bandwidth among the bandwidths in
	the given Array{<:Real, 1}
- `bias_correction`::Bool : If `true`, Calonico et al. (2014) bias correction is used
- `np_options`::Dict : Nonparametric estimation options, 
	Dict(:kernel => triangular_kernel, :poldegree => 2),
	where :kernel is function in `kernels.jl`,  :poldegree is `Int64`, 
	the polynomial degree in the local polynomial regression
- `lscv_options`::Dict : Least squares cross-validation options, used if and only if
	`bandwidth` is Array{T, 1}. Dictionary with keys :subsampling : if true only a random
	subsample (with replacement) of size :subsamplesize is used for the cross-validation;
	:window is either a Tuple (r_low, r_high) or :median. If a Tuple, then only observations
	``i`` such that ``x_i`` in interval [cutoff-r_low, cutoff+r_high] are used. If :median, only
	observations ``i`` such that ``x_i`` in interval 
	[median of ``x``s below cutoff, median of ``x``s above cutoff] are used.

##### Returns
- `tau_hat`::Float64 : Estimated sharp regression discontinuity design estimand

##### Examples
```julia

```

##### References
Calonico, Cattaneo, and Titiunik (2014): Robust Nonparametric Confidence Intervals
	for Regression-Discontinuity Designs. Econometrica, Vol. 82, No. 6.
"""
function rdd_sharpestimator(m::RDDModel,
							assignment_rule::Symbol,
							bandwidth::Union{T where T<:Real, Array{<:Real, 1}},
							bias_correction::Bool = true,
							np_options::Dict = Dict(:kernel => triangular_kernel,
													:poldegree => 1),
							lscv_options::Dict = Dict(:subsampling => true,
													  :subsamplesize => round(Int, length(m.y) / 3),
													  :window => :median))
	# check assignment rule
	if assignment_rule == :above_cutoff
		d_implied = Int.(m.x .>= m.cutoff)
	elseif assignment_rule == :below_cutoff
		d_implied = Int.(m.x .<= m.cutoff)
	else
		error("`assignment_rule` must be either `:above_cutoff` or `:below_cutoff`")
	end
	# check design
	if d_implied != m.d
		@warn("treatment participation data, `m.d`, is not consistent with sharp design " *
		 	  "or `assignment_rule`. Either the design is not sharp but fuzzy, or " *
		 	  "the `assignment_rule` is misspecified.")
	end
	# check np_options
	if haskey(np_options, :kernel) == false
		error("`np_options` must have key :kernel")
	elseif haskey(np_options, :poldegree) == false
		error("`np_options` must have key :poldegree")
	end
	if isa(np_options[:kernel], Function) == false
		error("`np_options[:kernel]` must be Function")
	elseif isa(np_options[:poldegree], Int) == false
		error("`np_options[:poldegree]` must be Integer")
	end
	# check lscv_options if cross validation is required
	cross_validation = ndims(bandwidth) != 0
	if cross_validation
		if haskey(lscv_options, :subsampling) == false
			error("`lscv_options` must have key :subsampling")
		elseif haskey(lscv_options, :subsamplesize) == false
			error("`lscv_options` must have key :subsamplesize")
		elseif haskey(lscv_options, :window) == false
			error("`lscv_options` must have key :window")
		end
		if isa(lscv_options[:subsampling], Bool) == false
			error("`lscv_options[:subsampling]` must be Bool")
		elseif isa(lscv_options[:subsamplesize], Int)
			error("`lscv_options[:subsamplesize]` must be Integer")
		end
		if isa(lscv_options[:window], Tuple)
			if length(lscv_options[:window]) != 2
				error("length(lscv_options[:window]) must be 2")
			end
		elseif lscv_options[:window] != :median
				error("`lscv_options[:window]` must be either a 2-long Tuple or :median")
		end
	end

	
	# sample size
	n = size(m.y)[1]
	# nonparametric options and bias correction
	kernel = np_options[:kernel]
	if bias_correction
		poldegree = np_options[:poldegree] + 1
	else
		poldegree = np_options[:poldegree]
	end

	# optimal bandwidth: scalar given bandwidth
	if !cross_validation
		h_opt = bandwidth
	# perform cross validation to find optimal bandwidth in `bandwidth` 
	# using original polynomial degreee
	else
		h_opt = _lscv_sharprdd(m, bandwidth, np_options[:poldegree], kernel, lscv_options)
	end

	# compute the (bias-corrected) estimate
	# estimate E[Y(1)|X=cutoff]
	muhat_t = localpoly_regression(x0=m.cutoff,
								   y=m.y[m.d .== 1],
								   x=m.x[m.d .== 1],
								   bandwidth=h_opt,
								   poldegree=poldegree,
								   kernel=kernel)
    # estimate E[Y(0)|X=cutoff]
	muhat_c = localpoly_regression(x0=m.cutoff,
								   y=m.y[m.d .== 0],
								   x=m.x[m.d .== 0],
								   bandwidth=h_opt,
								   poldegree=poldegree,
								   kernel=kernel)
	# rdd estimand
	tau_hat = muhat_t - muhat_c
	return tau_hat
end


"""
     _lscv_sharprdd(m::RDDModel,
						bandwidth::Array{<:Real, 1},
						poldegree::Int64,
						kernel::Function,
						lscv_options::Dict)

Finds optimal bandwidth for sharp regression discontinuity design with
leave-one-out least squares cross-validation.

If subsampling is specified in `lscv_options`, a subsample with replacement is drawn,
and LSCV is performed using only that subsample.

Only (sub)sample observations in a window around `m.cutoff` are used for loss-evaluation.
However, all the observations in the subsample are used for estimation
(except the left-one-out).

##### Arguments
- `m`::RDDModel : RDDModel type
- `bandwidth`::Array{<:Real, 1} Optimal bandwidth is searched among elements of this Array
- `poldegree`::Int64: Polynomial degree in the local polynomial regression
- `kernel`::Function : Kernel function in `kernels.jl`
- `lscv_options`::Dict : Least squares cross-validation options. Dictionary with keys
 	:subsampling : if true only a random subsample (with replacement) of size :subsamplesize
 	is used for the cross-validation; :window is either a Tuple (r_low, r_high) or :median.
 	If a Tuple, then only observations ``i`` such that ``x_i`` in interval
 	[cutoff-r_low, cutoff+r_high] are used. If :median, only observations ``i`` such that
 	``x_i`` in interval [median of ``x``s below cutoff, median of ``x``s above cutoff]
 	are used

##### Returns
-`h_opt`::typeof(bandwidth[1]) : LSCV-optimal bandwidth in `bandwidth`
"""
function _lscv_sharprdd(m::RDDModel,
						bandwidth::Array{<:Real, 1},
						poldegree::Int64,
						kernel::Function,
						lscv_options::Dict)
	n = length(m.y)
	# subsample data with replacement if required
	if lscv_options[:subsampling]
		ss_idx = rand(1:n, lscv_options[:subsamplesize])
		y = m.y[ss_idx]
		x = m.x[ss_idx]
		d = m.d[ss_idx]
	else
		y = m.y
		x = m.x
		d = m.d
	end
	n_s = length(y)  # = lscv_options[:subsamplesize] if subsampling true, = n oterwise

	# apply window for loss-evaluation
	if isa(lscv_options[:window], Tuple)
		r_low = lscv_options[:window][1]
		r_high = lscv_options[:window][2]
		window =  (m.cutoff - r_low .<= x) .* (x .<= m.cutoff + r_high)
	else
		a = StatsBase.median(x[x .<= m.cutoff])
		b = StatsBase.median(x[x .>= m.cutoff])
		window = (a .<= x) .* (x .<= b)
	end
	y_w = y[window]
	d_w = d[window]
	x_w = x[window]
	n_w = length(y_w)

	# find optimal bandwidth
	l2loss(y::Array{<:Real, 1}, yhat::Array{<:Real, 1}) = (y .- yhat)' * (y .- yhat)
	sse_lst = zeros(length(bandwidth))  # sum of squared errors
	h_idx = 0
	h_opt = bandwidth[1]  # initial optimal value
	# subsample idx of observations in the window, for leave-one-out cross validation
	ssidx_w = collect(1:n_s)[window]
	# loop through all bandwidths
	for h in bandwidth
		h_idx += 1  # update h idx
		yhat_w = zeros(n_w)  # pre-allocate predicted values
		# loop through all observations in the cross-val window
		for i in 1:n_w
			# mask for leave-one-out cross validation...
			mask = Bool.(true * ones(n_s))
			# ...leave out i in the window from the used subsample data
			mask[ssidx_w[i]] = false
			# if i is below cutoff, use only below-cutoff (subsample) data to predict y
			if x_w[i] <= cutoff
				mask = mask .* (x_w .<= cutoff)
			# if i is above cutoff, use only above-cutoff (subsample) data to predict y
			else
				mask = mask .* (x_w .>= cutoff)
			end
			# predict y
			mhat = localpoly_regression(x0=x_w[i],
										y=y[mask],
										x=x[mask],
										bandwidth=h,
										poldegree=poldegree,
										kernel=kernel)
			yhat_w[i] = mhat
		end
		# compute the loss for given h
		loss = l2loss(y_w, yhat_w)
		sse_lst[h_idx] = loss
		# update best h
		h_opt = sum(loss .<= sse_lst) == h_idx ? h_opt = h : h_opt = h_opt
	end
	return h_opt
end