# SyntheticAnalysisParameterEstimation_Bjordalsbakke2020

This repository contains code for sensitivity analysis, data generation, parameter estimation and presentation of results for the journal paper submission entitled "Parameter Estimation for Closed-Loop Lumped Parameter Models Using Synthetic Data".

The parameter estimation scripts describe 10 different estimation scenarios. Sett the parameter ps equal to the number of most sensitive parameters to include in the estimated subset.

The bar plot files assume that the results from subsets with 1 through 9 or 8 estimated parameters are found in a folder in the current directory named "1T" - "9T" for time series scripts, and "1S" - "9S" for clinical index based cost functions.

The script "scipy_VeWK3.py" contains the model implementation.

The script "timeout.py" contains a utility function.
