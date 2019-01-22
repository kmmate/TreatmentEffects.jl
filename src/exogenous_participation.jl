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
Bootstrap uses resampling with replacement. 
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
	bs_dist = @distributed (vcat) for rep in 1:bs_reps
		# bootstrap resampling with replacement
		bs_idx = rand(1:n, n)
		epm = ExogenousParticipationModel(m.y[bs_idx], m.d[bs_idx])
		ate_estimator(epm)
	end
end


"""
    bootstrap_htest(m::ExogenousParticipationModel, ate_0::Float64; bs_reps::Int = 999)

Test H_0: ATE = `ate_0` against H_1: ATE != `ate_0`, using bootstrap based distribution
of the average treatment effect estimator.
Bootstrap uses resampling with replacement. 
See [`ate_estimator`](@ref).

##### Arguments
- `m`::ExogenousParticipationModel : ExogenousParticipationModel model type
- `ate_0`::Float64 : H_0: ATE = `ate_0`
- `bs_reps`::Int : Number of bootstrap repetitions

##### Returns
- `p_value`::Float64 : empirical p-value

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("observational_data.csv")
y = data[:outcome]
d = data[:treatment_participation]
epm = ExogenousParticipationModel(y, d)
# test H_0: ATE = 2. against H_1: ATE != 2.
bootstrap_pvalue = bootstrap_htest(epm, 2.)
```
"""
function bootstrap_htest(m::ExogenousParticipationModel, ate_0::Float64; bs_reps::Int = 999)
	n = length(m.y)
	ate_hat = ate_estimator(m)
	test_stat = sqrt(n) * (ate_estimator - ate_0)
	# bootstrap distribution of ate_hat
	atehat_bsdistribution = bootstrap_distribution(m)
	# transform it to bootstrap distribution of test_stat
	teststat_bsdistribution = @. sqrt(n) * (atehat_bsdistribution - ate_0)
	# number of values more extreme than observed test_stat
	rejections = sum(abs(teststat_bsdistribution) .> abs(test_stat))
	# normalise to obtain empirical p-value
	p_value = rejections / bs_reps
	return p_value
end