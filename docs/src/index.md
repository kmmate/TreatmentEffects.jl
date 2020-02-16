# TreatmentEffects.jl

This is the documentation of the Julia package TreatmentEffects.jl. 

The package implements estimators and inference tools in the framework of the Rubin Causal Model,
the standard model for treatment effect evaluation using potential outcomes.  

TreatmentEffects.jl is only suitable for evaluation of a binary treatment.

TreatmentEffects.jl defines a type hierarchy of models which reflects the assumptions on the them,
and the nature of data (obervational vs. randomised control trial).

The type hierarchy provides a guidance on choosing the right model. See Getting started.

## Feature Highlights
* Observational data: matching estimator, regression discontinuity design
* Randomised experiment data: Fisher exact tests, instrumental variables
* Instrumental variables under violated stable unit treatment value assumption (SUTVA)

## Package Manual
```@contents
Pages = ["man/getting_started.md"]
Depth = 2
```

## API
```@contents
Pages = ["lib/types.md", "lib/functions.md"]
Depth = 2
```

## Index
```@index
```