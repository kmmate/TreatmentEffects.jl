#=

@author: Mate Kormos
@date: 05-Feb-2019

Define data cleaning methods
=#

"""
    drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1})

Drop rows with missing observation.
Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` is a missing value.

###### Arguments
- `y`::Array{<:Union{Missing, Any}, 1} : Outcome variable
- `d`::Array{<:Union{Missing, Any}, 1} : Treatment participation

##### Returns
- `y_dropped`::Array{<:Any, 1} : Outcome variable with missing rows dropped
- `d_dropped`::Array{<:Any, 1} : Treatment participation with missing rows dropped

##### Examples
```julia-repl
julia> y = [1, 2, missing];
julia> d = [4, missing, 5];
julia> drop_missing(y, d)
[1], [4]
```
"""
function drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1})
	# indices with not missing entries
	notmissing_idx = (isequal.(y, missing) .== false) .* (isequal.(d, missing) .== false)
	y_dropped_uniontype = y[notmissing_idx]
	d_dropped_uniontype = d[notmissing_idx]
	notmissing_length = sum(notmissing_idx)
	y_dropped, d_dropped = zeros(notmissing_length), zeros(notmissing_length)
	for i in 1:notmissing_length
		y_dropped[i] = y_dropped_uniontype[i]
		d_dropped[i] = d_dropped_uniontype[i]
	end
	return y_dropped, d_dropped
end


"""
    drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1},
	x::Array{<:Union{Missing, Any}, 1})

Drop rows with missing observation.
Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` or ``x[i]`` is a missing value.

###### Arguments
- `y`::Array{<:Union{Missing, Any}, 1} : Outcome variable
- `d`::Array{<:Union{Missing, Any}, 1} : Treatment participation
- `x`::Array{<:Union{Missing, Any}, 1} : Covariate

##### Returns
- `y_dropped`::Array{<:Any, 1} : Outcome variable with missing rows dropped
- `d_dropped`::Array{<:Any, 1} : Treatment participation with missing rows dropped
- `x_dropped`::Array{<:Any, 1} : Covariate with missing rows dropped

##### Examples
```julia-repl
julia> y = [1, 2, missing, 8, 889];
julia> d = [4, missing, 5, 9, 789];
julia> x = [10, 12, 18, missing, 666];
julia> y, d, x = drop_missing(y, d, x);
julia> println(x)
[10, 666]
```
"""
function drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1},
	x::Array{<:Union{Missing, Any}, 1})
	# indices with not missing entries
	notmissing_idx = (isequal.(y, missing) .== false) .* (isequal.(d, missing) .== false) .*
	 (isequal.(x, missing) .== false)
	y_dropped_uniontype = y[notmissing_idx]
	d_dropped_uniontype = d[notmissing_idx]
	x_dropped_uniontype = x[notmissing_idx]
	notmissing_length = sum(notmissing_idx)
	y_dropped = zeros(notmissing_length)
	d_dropped = zeros(notmissing_length)
	x_dropped = zeros(notmissing_length)
	for i in 1:notmissing_length
		y_dropped[i] = y_dropped_uniontype[i]
		d_dropped[i] = d_dropped_uniontype[i]
		x_dropped[i] = x_dropped_uniontype[i]
	end
	return y_dropped, d_dropped, x_dropped
end



"""
    drop_missing(y::Array{<:Union{Missing, Any}, 1},
		d::Array{<:Union{Missing, Any}, 1},
 		x::Array{<:Union{Missing, Any}, 2})

Drop rows with missing observation.
Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` or ``x[i, k]`` is a missing value for any ``k``.

###### Arguments
- `y`::Array{<:Union{Missing, Any}, 1} : Outcome variable
- `d`::Array{<:Union{Missing, Any}, 1} : Treatment participation
- `x`::Array{<:Union{Missing, Any}, 2} : Covariates, each row corresponds to an oberservation

##### Examples
```julia-repl
julia> y = [1, 2, missing, 8]
julia> d = [4, missing, 5, 9]
julia> x = [20 30 40; 90 80 100; 200 278 29; missing 900 8897]
julia> drop_missing!(y, d, x)
julia> println(x)
[20 30 40]
```
"""
function drop_missing(y::Array{<:Union{Missing, Any}, 1},
	d::Array{<:Union{Missing, Any}, 1},
	x::Array{<:Union{Missing, Any}, 2})
	# indices with not missing entries
	# check y and d
	notmissing_idx = (isequal.(y, missing) .== false) .* (isequal.(d, missing) .== false)
	# check x's
	for col in 1:size(x)[2]
		notmissing_idx = notmissing_idx .* (isequal.(x[:, col], missing) .==false)
	end
	y_dropped_uniontype = y[notmissing_idx]
	d_dropped_uniontype = d[notmissing_idx]
	x_dropped_uniontype = x[notmissing_idx, :]
	notmissing_length = sum(notmissing_idx)
	y_dropped = zeros(notmissing_length)
	d_dropped = zeros(notmissing_length)
	x_dropped = zeros(notmissing_length, size(x)[2])
	for i in 1:notmissing_length
		y_dropped[i] = y_dropped_uniontype[i]
		d_dropped[i] = d_dropped_uniontype[i]
		for col in 1:size(x)[2]
			x_dropped[i, col] = x_dropped_uniontype[i, col]
		end
	end
	return y_dropped, d_dropped, x_dropped
end