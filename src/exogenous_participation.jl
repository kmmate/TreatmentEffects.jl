#=

@author: Mate Kormos
@date: 08-Jan-2019

Defines the methods of the primitive composite type ExogenousParticipationModel <: ObservationalModel
=#

"""
    ate_estimator(m::ExogenousParticipationModel)

Estimate the average treatment effect (ATE).

Under the assumption that treatment participation, ``D`` is independent of potential outcomes ``[Y(0) Y(1)]`` 
the estimator is consistent and unbiased for ATE (which in this case coincides with the average treatment effect for the treated, ATT).

##### Arguments
-`m::ExogenousParticipationModel` : ExogenousParticipationModel model type

##### Returns
- `ate_hat`: estimted ATE

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("observational_data.csv")
y = data[:outcome]
d = data[:treatment_participation]
epm = ExogenousParticipationModel(y, d)
ate_hat = ate_estimator(epm)
```
"""
function ate_estimator(m::ExogenousParticipationModel)
	n_treated = sum(m.d)
	n_control = length(m.y) - n_treated
	mean_treated = sum(m.y[m.d .== 1]) / n_treated
	mean_control = sum(m.y[m.d .== 0]) / n_control
	return mean_treated - mean_control
end