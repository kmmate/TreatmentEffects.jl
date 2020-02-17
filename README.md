# TreatmentEffects.jl

[travis-url]: https://travis-ci.com/kmmate/TreatmentEffects.jl 
[travis-img]: https://travis-ci.com/kmmate/TreatmentEffects.jl.svg?branch=master

[![Build Status][travis-img]](travis-url)

**TreatmentEffects** implements estimators and inference tools in the framework of the Rubin Causal Model,
the standard model for treatment effect evaluation using potential outcomes.

The package is only suitable for evaluation of a binary treatment,
using cross-sectional data.

Type hierarchy  (see Getting started) provides a guidance on choosing the right model,
and the associated estimators.

## Implemented
* Observational data: matching estimator, (nonparametric) regression discontinuity design

## TODO
* Randomised experiment data: Fisher exact tests, instrumental variables
* Instrumental variables under violated stable unit treatment value assumption (SUTVA)

## References
Rubin, Donald (1974). "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies". Journal of Educational Psychology 66 (5): 688–701
