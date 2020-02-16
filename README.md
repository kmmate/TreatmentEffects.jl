# TreatmentEffects.jl

** TreatmentEffects ** implements estimators and inference tools in the framework of the Rubin Causal Model,
the standard model for treatment effect evaluation using potential outcomes.

The package is only suitable for evaluation of a binary treatment,
using cross-sectional data.

Type hierarchy  (see Getting started) provides a guidance on choosing the right model,
and the associated estimators.

## Implemented
* Observational data: matching estimator, regression discontinuity design

## TODO
* Randomised experiment data: Fisher exact tests, instrumental variables
* Instrumental variables under violated stable unit treatment value assumption (SUTVA)