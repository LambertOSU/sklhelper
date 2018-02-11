# sklhelper

This repository contains helper classes for the initial evaluation of scikit-learn predictors.  It is based on a previously constructed Notebook and is currently in development.

**sklhelpClassify** - This class imports a pandas DataFrame and performs a *k*-fold validation test on various scikit-learn classifiers using predefined parameters and reports ranked results.

Method | Description
--- | ---
get_data(df) | Imports a pandas DataFrame.
set_target(name) | Defines the name of the column to be predicted.
kfold() | Runs a k-fold validation test.  Default k=5.
ranked_summary() | Reports mean, median, and standard deviation for all models, ranked by mean.
report() | View all simulation results.
