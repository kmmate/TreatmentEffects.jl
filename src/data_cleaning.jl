#=

@author: Mate Kormos
@date: 05-Feb-2019

Define data cleaning methods
=#

"""
    drop_missing!(y::Array{Any, 1}, d::Array{Any, 1})

Drop rows with missing observation.
Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` is a missing value.

###### Arguments
- `y`::Array{Any, 1} : Outcome variable
- `d`::Array{Any, 1} : Treatment participation

##### Examples
```julia-repl
julia> y = [1, 2, missing]
julia> d = [4, missing, 5]
julia> drop_missing!(y, d)
julia> println(y)
[1]
```
"""
function drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1})
	# indices with not missing entries
	not_missing_idx = (isequal.(y, missing) .== false) .* (isequal.(d, missing) .== false)
	y_dropped_uniontype = y[not_missing_idx]
	d_dropped_uniontype = d[not_missing_idx]
	y_dropped, d_dropped = zeros(sum(not_missing_idx)), zeros(sum(not_missing_idx))
	for i in 1:length(y_dropped)
		y_dropped[i] = y_dropped_uniontype[i]
		d_dropped[i] = d_dropped_uniontype[i]
	end
	return y_dropped, d_dropped
end


"""
    drop_missing!(y::Array{Any, 1}, d::Array{Any, 1},
     x::Array{Any, 1})

Drop rows with missing observation.
Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` or ``x[i]`` is a missing value.

###### Arguments
- `y`::Array{Any, 1} : Outcome variable
- `d`::Array{Any, 1} : Treatment participation
- `x`::Array{Any, 1} : Covariate

##### Examples
```julia-repl
julia> y = [1, 2, missing, 8]
julia> d = [4, missing, 5, 9]
julia> x = [10, 12, 18, missing]
julia> drop_missing!(y, d, x)
julia> println(x)
[10]
```
"""
#=
function drop_missing(y::Array{Any, 1}, d::Array{Any, 1}, x::Array{Any, 1})
	# indices with not missing entries
	not_missing_idx = (isequal.(y, missing) .== false) .* (isequal.(d, missing) .== false) .*
	 (isequal.(x, missing) .== false)
	y = y[not_missing_idx]
	d = d[not_missing_idx]
	x = x[not_missing_idx]
	return y, d, x
end

"""
    drop_missing!(y::Array{Any, 1}, d::Array{Any, 1}, x::Array{Any})

Drop rows with missing observation.
Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` or ``x[i, k]`` is a missing value for any ``k``.

###### Arguments
- `y`::Array{Any, 1} : Outcome variable
- `d`::Array{Any, 1} : Treatment participation
- `x`::Array{Any} : Covariates, each row corresponds to an oberservation

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
function drop_missing(y::Array{Any, 1}, d::Array{Any 1}, x::Array{Any, 1})
	# indices with not missing entries
	# check y and d
	not_missing_idx = (isequal.(y, missing) .== false) .* (isequal.(d, missing) .== false)
	# check x's
	for col in 1:size(x)[2]:
		not_missing_idx = not_missing_idx .* (isequal.(x[:, col], missing) .==false)
	end
	y = y[not_missing_idx]
	d = d[not_missing_idx]
	x = x[not_missing_idx, :]
	return y, d, x
end
=#