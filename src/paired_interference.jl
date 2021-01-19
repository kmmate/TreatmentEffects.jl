#=

@author: Mate Kormos

Defines the methods of the primitive composite type
PairedInterferenceModel <: InterferenceModel

=#

using Distributed

"""
    late_estimator(m::PairedInterferenceModel)

Estimate local averate treatment effects (LATE) for compliers:
``𝔼[Y(D_A=1, D_B=0)-Y(D_A=0, D_B=0)| D_A(Z_A=1,D_B=1)=1, D_A(Z_A=1,Z_B=0)=1]`` and
``𝔼[Y(D_A=0, D_B=1)-Y(D_A=0, D_B=0)| D_A(Z_A=1,D_B=1)=1, D_A(Z_A=0,Z_B=1)=1]``.
The estimator is consistent under usual instrumental variables assumptions extended for interference.
See Kormos and Lieli.

##### Arguments
-`m`::PairedInterferenceModel : Paired Interference model type

##### Returns
-`late_hat`::Array{<:Real, 1} : Estimate of ``[𝔼[Y(D_A=0,D_B=0)], 𝔼[Y(D_A=1, D_B=0)-Y(D_A=0, D_B=0)| D_A(Z_A=1,D_B=1)=1, D_A(Z_A=1,Z_B=0)=1], 𝔼[Y(D_A=0, D_B=1)-Y(D_A=0, D_B=0)| D_A(Z_A=1,D_B=1)=1, D_A(Z_A=0,Z_B=1)=1], ν]`` where ``ν`` is a spill-over like effect.

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("pim_data.csv")
y = data[:outcome]
d_a = data[:treatment_takeup_a]  # d_a_i=1 iff i'th member-A is treated
d_b = data[:treatment_takeup_b]  # d_b_i=1 iff i'th member-B is treated
z_a = data[:treatment_assignment_a]  # z_a_i=1 iff i'th member-A is assigned to treated
z_b = data[:treatment_assignment_b]  # z_b_i=1 iff i'th member-B is assigned to treated
pim = PairedInterferenceModel(y, d_a, d_b, z_a, z_b)
late_hat = late_estimator(pim)
```

##### References
Kormos and Lieli, "Treatment Effect Analysis for Pairs with
Endogenous Treatment Take-up" (forthcoming)

"""
function late_estimator(m::PairedInterferenceModel)
    n = length(m.y)
    d = hcat(ones(n,1), m.d_a, m.d_b, m.d_a .* m.d_b)
    z = hcat(ones(n,1), m.z_a, m.z_b, m.z_a .* m.z_b)
    (z'*d)\(z'*m.y)
end

"""
    bootstrap_distribution(m::PairedInterferenceModel)

Return bootstrap distribution of the local average treatment effect estimator.
See [`late_estimator`](@ref).
Bootstrap uses resampling with replacement. 

##### Arguments
- `m`::PairedInterferenceModel : PairedInterferenceModel model type
- `bs_reps`::Int : Number of bootstrap repetitions

##### Returns
- `dist`::Array{Float64, 2} : `length(late_estimator(m))`-by-`bs_reps` array,
    each column corresponds to a bootstrap sample 

##### Examples
```julia
using TreatmentEffects, CSV
data = read_csv("pim_data.csv")
y = data[:outcome]
d_a = data[:treatment_takeup_a]  # d_a_i=1 iff i'th member-A is treated
d_b = data[:treatment_takeup_b]  # d_b_i=1 iff i'th member-B is treated
z_a = data[:treatment_assignment_a]  # z_a_i=1 iff i'th member-A is assigned to treated
z_b = data[:treatment_assignment_b]  # z_b_i=1 iff i'th member-B is assigned to treated
pim = PairedInterferenceModel(y, d_a, d_b, z_a, z_b)
# asymptotic bootsrap variance
bs_dist = bootstrap_distribution(pim)
n = length(y)
asym_var = var(sqrt(n) * (bs_dist - mean(bs_dist)))
```

"""
function bootstrap_distribution(m::PairedInterferenceModel; bs_reps::Int = 999)
    n = length(m.y)
    bs_dist = @distributed (hcat) for rep in 1:bs_reps
        bs_idx = rand(1:n, n)
        pim = PairedInterferenceModel(m.y[bs_idx], m.d_a[bs_idx], m.d_b[bs_idx],
                                      m.z_a[bs_idx], m.z_b[bs_idx])
        late_estimator(pim)
    end
end