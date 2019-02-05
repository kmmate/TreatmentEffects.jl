#=
Tests for data_celaning.jl

=#


# testing drop_missing!
@testset "drop_missing!" begin
	# testing drop_missing!(y::Array{<:Real, 1}, d::Array{<:Real, 1})
	y = [1, 2, missing]
	d = [4, missing, 5]
	y, d = drop_missing(y, d)
	println(y, d)
	@test y == [1] && d == [4]
end