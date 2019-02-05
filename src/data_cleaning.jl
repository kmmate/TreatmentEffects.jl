#=

@author: Mate Kormos
@date: 05-Feb-2019

Define data cleaning methods
=#

"""

Drop rows with missing observation.

Row ``i`` is dropped whenever ``y[i]`` or ``d[i]`` is a missing value.
"""
function drop_missing!(y::Array{<:Real, 1}, d::Array{<:Real, 1})
	# indices with not missing entries
	not_missing_idx = isequal.(y, missing) .== false && isequal.(d, missing) .== false
	y = y[not_missing_idx]
	d = d[not_missing_idx]
	#return y, d
end