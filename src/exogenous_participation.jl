#=

@author: Mate Kormos
@date: 08-Jan-2019

Defines the methods of the primitive composite type
ExogenousParticipationModel <: ObservationalModel

=#

using Distributed

"""
    ate_estimator(m::ExogenousParticipationModel)

Estimate the average treatment effect (ATE).

Under the assumption that treatment participation, ``D`` is independent of potential
outcomes ``[Y(0) Y(1)]``, the estimator is consistent and unbiased for ATE
(which in this case coincides with the average treatment effect for the treated, ATT).

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


"""
    bootstrap_distribution(m::ExogenousParticipationModel; bs_reps::Int = 999)

Return bootstrap distribution of the average treatment effect estimator. 
See [`ate_estimator`](@ref).

##### Arguments
- `m`::ExogenousParticipationModel : ExogenousParticipationModel model type
- `bs_reps`::Int : Number of bootstrap repetitions

##### Returns
- `dist`::Array{Float64} : `bs_reps`-long array,
	each entry corresponds to a bootstrap sample

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("observational_data.csv")
y = data[:outcome]
d = data[:treatment_participation]
n = length(y)
epm = ExogenousParticipationModel(y, d)
# get asymptotic variance
bs_dist = bootstrap_distribution(epm)
asym_var = var(sqrt(n) * (bs_dist - mean(bs_dist)))
```
"""
function bootstrap_distribution(m::ExogenousParticipationModel; bs_reps::Int = 999)
	n = length(m.y)
	#bs_dist = Array{Float64}(undef, bs_reps)
	bs_dist = @distributed (vcat) for rep in 1:bs_reps
		# bootstrap resampling
		bs_idx = rand(1:n, n)
		epm = ExogenousParticipationModel(m.y[bs_idx], m.d[bs_idx])
		ate_estimator(epm)
	end
end