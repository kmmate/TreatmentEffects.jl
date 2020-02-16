#=
Tests for data_celaning.jl

=#


# testing drop_missing
@testset "drop_missing" begin
	# testing drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1})
	y = [1, 2, missing, 7]
	d = [4, missing, 5, 30.]
	y_dropped, d_dropped = drop_missing(y, d)
	@test y_dropped == [1, 7] && d_dropped == [4, 30]
	
	# testing drop_missing(y::Array{<:Union{Missing, Any}, 1}, d::Array{<:Union{Missing, Any}, 1},
	# x::Array{<:Union{Missing, Any}, 1})
	y = [1, 2, missing, 7]
	d = [4, missing, 5, 30.]
	x = [87, 23, 87, missing]
	y_dropped, d_dropped, x_dropped = drop_missing(y, d, x)
	@test y_dropped == [1] && d_dropped == [4] && x_dropped == [87]

	# testing drop_missing(y::Array{<:Union{Missing, Any}, 1},
	#	d::Array{<:Union{Missing, Any}, 1},
 	#	x::Array{<:Union{Missing, Any}, 2})
	y = [1, 2, missing, 8]
	d = [4, missing, 5, 9]
	x = [20 30 40; 90 80 100; 200 278 29; missing 900 8897]
	y_dropepd, d_dropped, x_dropped = drop_missing(y, d, x)
	@test y_dropped == [1] && d_dropped == [4] && x_dropped == [20 30 40]
end