# Getting started

## Installation

```julia-repl
(v1.0) pkg> add https://github.com/kmmate/TreatmentEffects.jl
```

## Model choice

As of yet only cross sectional data models are implemented. The right
model can be chosen by the following decision tree of assumptions.

Only the models at terminal branches can be instantiated; higher level branches
indicate abstract types.

* If **SUTVA** (Stable Unit Treatment Value Assumption) holds: [`SUTVAModel`](@ref)
	
	SUTVA means that the potential outcome of unit ``i`` only depends on unit ``i``'s treatment.
	Formally, ``Y_i(D_i, D_j) = Y_i(D_i) \forall i,j:j\neq i``.
	
	* If **Observational data**: [`ObservationalModel`](@ref)

		By observational data we mean that the data do not come from a randomised control trial.
		The uncertainty in this case purely stems from sampling.

		* If **no self selection into treatment**: [`ExogenousParticipationModel`](@ref)

			By no self selection into treatment, we mean that participation in the treatment is
			independent from the potential outcomes. Formally ``[Y_i(0), Y_i(1)]`` independent of ``D_i``.

			Implemented methods: [`ate_estimator`](@ref), [`bootstrap_distribution`](@ref), [`bootstrap_test`](@ref)

			!!! warning

			This model type serves illustrative purposes. 
			Use with real data is not recommended as the required assumption is extremely strong.

		* If **CIA** (conditional independence, i.e. unconfundedness) holds: [`CIAModel`](@ref)

			By CIA we mean that conditional on a set of pre-treatment covariate(s) ``x``, the treatment
			is as good as random. Formally, given ``x_i`` ``[Y_i(0), Y_i(1)]`` is independent of ``D_i``.

			* If **cut-off based treatment participation rule** : [`RDDModel`](@ref)

				By cut-off based treatment participation we mean that ``D_i = 1`` if and only if
				``x_i`` exceeds a certain cut-off value. This gives rise to the Regression Discontinuity Design.

				Implemented methods: 

			* Else **no known cut-off based treatment participation rule** : [`MatchingModel`](@ref)

				The most general case of `CIAModel`, which is not an `RDDModel`.

				Implemented methods: 

	* Else **Randomised Control Trial data**: [`RCTModel`](@ref)

		By randomised control trial data we mean that the data come from a randomised control trial,
		where the treatment assignment is randomised.
		The uncertainty in this case stems from the treatment assignment and/or sampling.

		* If **all units perfectly comply with their assignment**: [`PerfectComplianceModel`](@ref)
		
		* Else **some units do not comply with their treatment assigment**: [`ImperfectComplianceModel`](@ref)
			
			* If **non-compliance with assigment is exogenous**: [`ExogenousNoncomplianceModel`](@ref)

			!!! warning

			Usage not recommended because of strict assumptions.

			* Else **non-compliance with assigment is endogenous**: [`EndogenousNoncomplianceModel`](@ref)

				Implemented methods:

* Else **SUTVA violated**: `InterferenceModel`

	Rubin Causal models where SUTVA is violated have just been gaining ground recently.
	Here the experimental data based instrumental variable estimator is implemented
	for paired interference, based on Kormos(2019). 