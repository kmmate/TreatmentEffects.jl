#=

@author: Mate Kormos
@date: 08-Jan-2019

Provides the hierarchy of abtract and primitve model types for treatment effect estimation.
=#


# highest level
"""
Cross sectional data model with independent observations.
"""
abstract type CrossSectionModel end


# CrossSectionModel subtypes
"""
Stable Unit Treatment Value Assumption (SUTVA) holds.
"""
abstract type SUTVAModel <: CrossSectionModel end  

"""
Stable Unit Treatment Value Assumption (SUTVA) is violated. Only available for paired interference.
"""
abstract type InterferenceModel <: CrossSectionModel end


#	SUTVAModel subtypes
"""
Data comes from a randomised control trial (RCT).
"""
abstract type RCTModel <: SUTVAModel end

"""
Observational data, self-selection into treatment.
"""
abstract type ObservationalModel <: SUTVAModel end


#		RCTModel subtypes
"""
Units perfectly comply with their treatment assignment, i.e. treatment takeup equal to treatment assignment.
That is ``D(1) = 1`` iff ``Z = 1`` and ``D(0) = 0`` iff ``Z = 0``.

##### Fields
- `y` : Observed outcome of interest
- `z` : Treatment assignment

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("experimental_data.csv")
y = data[:outcome]
z = data[:treatment_assignment]
pcm = PerfectComplianceModel(y, z)
```
"""
mutable struct PerfectComplianceModel <: RCTModel
	y::Array{<:Real, 1}
	z::Array{<:Real, 1}
    function PerfectComplianceModel(y::Array{<:Real, 1}, z::Array{<:Real, 1})  # inner constructor to check dimensions
    	if length(y) == length(z)
    		return new(y, z)
    	else
    		error("`y` and `z` must have the same number of observations")
    	end
    end
end

"""
Units do not comply perfectly with their treatment assignment, i.e. treatment takeup not always equal to treatment assignment.
That is ``P(D(1) = 1) != 1``.
"""  
abstract type ImperfectComplianceModel <: RCTModel end


#			ImperfectComplianceModel primitive subtypes
"""
Noncompliance with treatment assigment is exogenous.

##### Fields
- `y` : Observed outcome of interest
- `d` : Treatment takup

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("experimental_data.csv")
y = data[:outcome]
d = data[:treatment_takeup]
exnm = ExogenousNoncomplianceModel(y, d)
```
"""
mutable struct ExogenousNoncomplianceModel <: ImperfectComplianceModel
	y::Array{<:Real, 1}
	d::Array{<:Real, 1}
	function ExogenousNoncomplianceModel(y::Array{<:Real, 1}, d::Array{<:Real, 1}) 
		if length(y) == length(d)
			return new(y, d)
		else
			error("`y` and `d` must have the same number of observations")
		end
	end
end
"""
Noncompliance with treatment assigment may be endogenous.

##### Fields
- `y` : Observed outcome of interest
- `z` : Treatment assignment
- `d` : Treatment takup

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("experimental_data.csv")
y = data[:outcome]
z = data[:treatment_assignment]
d = data[:treatment_takeup]
encm = EndogenousNoncomplianceModel(y, z, d)
```
"""  
mutable struct EndogenousNoncomplianceModel <: ImperfectComplianceModel
	y::Array{<:Real, 1}
	z::Array{<:Real, 1}
	d::Array{<:Real, 1}
	function EndogenousNoncomplianceModel(y::Array{<:Real, 1}, z::Array{<:Real, 1}, d::Array{<:Real, 1})
		if length(y) == length(z) == length(d)
			return new(y, z, d)
		else
			error("`y`, `z`, `d` must have the same number of observations") 
		end
	end
end  


#		ObservationalModel primitive subtypes
"""
Potential outcomes independent of participation, that is ``[Y(0), Y(1)]`` is independent of ``D``.

It is assumed that the only source of uncertainty is sampling. That is, the observed dataset {(y_i, d_i)} i=1,...,n
is and i.i.d. sample from a population, and the experimenter has no control over treatment participation, D.

##### Fields
- `y` : Observed outcome of interest
- `d` : Treatment participation

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("observational_data.csv")
y = data[:outcome]
d = data[:treatment_participation]
epm = ExogenousParticipationModel(y, d)
```
"""
mutable struct ExogenousParticipationModel <: ObservationalModel
	y::Array{<:Real, 1}
	d::Array{<:Real, 1}
	function ExogenousParticipationModel(y::Array{<:Real, 1}, d::Array{<:Real, 1})
		if length(y) == length(d) && ndims(y) == ndims(d) == 1
			return new(y, d)
		else 
			error("`y` and `d` must be one dimensional and have the same number of observations")
		end
	end
end  

"""
Given a set of covariates, potential outcomes independent of participation, 
that is given ``x``, ``[Y(0), Y(1)]`` is independent of ``D``.

It is assumed that the only source of uncertainty is sampling. That is, the observed dataset {(y_i, d_i, x_i)} i=1,...,n
is and i.i.d. sample from a population, and the experimenter has no control over treatment participation, D.

##### Fields
- `y` : Observed outcome of interest
- `d` : Treatment participation
- `x` : Covariates

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("observational_data.csv")
y = data[:outcome]
d = data[:treatment_takeup]
x = data[:covariates]
ciam = CIAModel(y, d, x)
```
"""
mutable struct CIAModel <: ObservationalModel
	y::Array{<:Real, 1}
	d::Array{<:Real, 1}
	x::Array{<:Real}
	function CIAModel(y::Array{<:Real, 1}, d::Array{<:Real, 1}, x::Array{<:Real})
		if size(y)[1] == size(d)[1] == size(x)[1]
			return new(y, d, x)			
		else
			error("`y`, `d`, `x` must have the same number of observations") 
		end
	end
end