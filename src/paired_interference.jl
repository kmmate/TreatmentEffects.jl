#=

@author: Mate Kormos

Defines the methods of the primitive composite type
PairedInterferenceModel <: InterferenceModel

=#

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
