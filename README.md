# Montecarlo funcs

Repo to house my (currently) small collection of resampling scripts.
___
WORK IN PROGRESS
___

**classifier_bs_funcs.py**: Compute classification metrics and classification reports with CI.

**corr_perm_funcs.py**: Permute correlations and correct for multiple comparisons using the maxT method.

**diff_perm_funcs.py**: Use permutation tests to examine differences between groups. Currently supports t-tests (calling functions from scipy) and ANOVAs (using pyvttbl to support fixed, random and mixed tests).
NOTE: pyvttbl is no longer supported and **does not run** on Python3.
