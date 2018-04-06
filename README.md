# Montecarlo funcs

Repo to house my (currently) small collection of resampling scripts.

__WORK IN PROGRESS__

**bootstrap_funcs.py**: Currently houses model validation using bootstrap. Why use k folds when the model can be tested on multiple out of group / bag / sample / *your favourite word here* data and the range of possible performance can be determined?

**corr_perm_funcs.py**: Permute correlations and correct for multiple comparisons using the maxT method.

**diff_perm_funcs.py**: Use permutation tests to examine differences between groups. Currently supports t-tests (calling functions from scipy) and ANOVAs (using pyvttbl to support fixed, random and mixed tests).
