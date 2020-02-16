#=

@author: Mate Kormos
@date: 12-Feb-2019

Define kernels for nonparametric regressions.

All kernel functions must take the same inputs: 
point of evaluation and the order of kernel.

=#

"""
    uniform_kernel(u::Float64; kernelorder::Int64=2)

Uniform kernel.

###### Arguments
- `u`::Float64 : Point of evaluation
- `kernelorder`::Int64 : Order of kernel, must be 2

###### Returns
- `ku`::Float64 : Value of kernel at `u`
"""
function uniform_kernel(u::Float64; kernelorder::Int64=2)
	if kernelorder !=2
		error("for `uniform_kernel` only `kernelorder=2` is defined")
	end
	ku = 0.5 * Int(-1 <= u <= 1)
	return ku
end

"""
    triangular_kernel(u::Float64; kernelorder::Int64=2)

Triangular kernel.

###### Arguments
- `u`::Float64 : Point of evaluation
- `kernelorder`::Int64 : Order of kernel, must be 2

###### Returns
- `ku`::Float64 : Value of kernel at `u`
"""
function triangular_kernel(u::Float64; kernelorder::Int64=2)
	if kernelorder !=2
		error("for `triangular_kernel` only `kernelorder=2` is defined")
	end
	ku = (1 - abs(u)) * Int(-1 <= u <= 1)
	return ku
end

"""
    epanechnikov_kernel(u::Float64; kernelorder::Int64=2)

Epanechnikov kernel.

###### Arguments
- `u`::Float64 : Point of evaluation
- `kernelorder`::Int64 : Order of kernel, must be one of 2, 4, 6

###### Returns
- `ku`::Float64 : Value of kernel at `u`
"""
function epanechnikov_kernel(u::Float64; kernelorder::Int64=2)
	if (kernelorder in [2, 4, 6]) == false
		error("for `epanechnikov_kernel` only `kernelorder` in [2, 4, 6] is defined")
	elseif kernelorder == 2
		ku = 0.75 * (1 - u ^ 2) * Int(-1 <= u <= 1)
		return ku
	elseif kernelorder == 4
		c = 0.75 * (1 - u ^ 2) * Int(-1 <= u <= 1)
		ku = 15 / 8 * (1 - 7 / 3 * u ^ 2) * c
		return ku
	elseif kernelorder == 6
		c = 0.75 * (1 - u ^ 2) * Int(-1 <= u <= 1)
		ku = 175 / 64 * (1 - 6 * u ^ 2 + 33 / 5 * u ^ 4) * c
		return ku
	end
end

"""
    gaussian_kernel(u::Float64; kernelorder::Int64=2)

Gaussian kernel.

###### Arguments
- `u`::Float64 : Point of evaluation
- `kernelorder`::Int64 : Order of kernel, must be one of 2, 4, 6

###### Returns
- `ku`::Float64 : Value of kernel at `u`
"""
function gaussian_kernel(u::Float64; kernelorder::Int64=2)
	if (kernelorder in [2, 4, 6]) == false
		error("for `gaussian_kernel` only `kernelorder` in [2, 4, 6] is defined")
	elseif kernelorder == 2
		ku = (2 * pi) ^ (-0.5) * exp(-0.5 * u ^ 2)
		return ku
	elseif kernelorder == 4
		c = (2 * pi) ^ (-0.5) * exp(-0.5 * u ^ 2)
		ku = 0.5 * (3 - u ^ 2) * c
		return ku
	elseif kernelorder == 6
		c = (2 * pi) ^ (-0.5) * exp(-0.5 * u ^ 2)
		ku = 1 / 8 * (15 - 10 * u ^ 2 + u ^ 4) * c
		return ku
	end
end