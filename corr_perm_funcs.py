import scipy
import numpy as np
import joblib
import itertools
import matplotlib.pylab as plt

def perm_corr_single(dataframe, combos, perm_idx, method = scipy.stats.spearmanr):
    '''

    Generates a single iteration of permuated correlation values
    Used inside corr_mcc with perm_idx as the iterator.

    Input:
    dataframe - Dataframe to be correlated
    combos - List of combinations (from itertools.combinations) in dataframe to be used - Note: Label 1 will be permutated.
    perm_idx - List of indexes to use on each iteration
    method - Correlation method (spearman by default)

    Output:
    corrs - Single instance of permuted correlations (lower triangle)
    '''

    corrs = np.zeros(len(combos))

    for labels_n, labels in enumerate(combos):
        x = dataframe[labels[0]].values
        y = dataframe[labels[1]].loc[perm_idx].values
        corrs[labels_n] = method(x, y)[0]

    return(corrs)



def p_perm(obs_corrs, perm_corrs):
    '''
    Calculate p values and maxT corrected p values for correlations

    maxT correction collects the maximum single value across valiables for each iteration.
    The number of values from the maximum single value distribution greater than the observed
    values becomes the corrected P value against multiple comparisons.

    Input:
    obs_corrs - Observed correlations
    perm_corrs - Permuated correlations

    Output:
    p_raw - Uncorrected pvalues
    p_maxT - Corrected p values
    '''

    p_raw = np.zeros_like(obs_corrs)

    p_raw = np.array([sum(abs(perm_corrs[:, corr_n] > abs(obs_corrs[corr_n]))) / n_shuffles for corr_n in range(0, len(obs_corrs))])

    maxT = np.array([max(max_val) for max_val in perm_corrs])
    p_maxT = np.array([sum(abs(maxT) > abs(obs_corrs[corr_n])) / n_shuffles for corr_n in range(0, len(obs_corrs))])

    return(p_raw, p_maxT)


def corr_perm(dataframe,
             n_shuffles = 10000,
             seed = 1010,
             replace = False,
             n_jobs = 1,
             verbose = 10,
             method = scipy.stats.spearmanr):

    """
    Main function for running permutation / bootstrapping correlations

    Input:
    dataframe - Data (r * c) to be correlated in the form of a pandas dataframe
    n_shuffles - Number of permutations to run (Default - 10,000^)
    seed - Seed for reproducable result
    replace - Replace observation in shuffle? False = permutation, True = bootstrap
    n_jobs - Number of cores to use
    verbose - Level of verbosity (Default- 10 [very verbose])
    method - Correlation method from scipy.stats (Default - spearmanr)

    Output:
    obs_corrs - Observed correlation values
    perm_corrs - n_shuffles * lower triangle correlation matrix

    ^ 10,000 permutations of a 16 * 57 matrix takes ~ 40 minutes using 8 cores of
    a 3.6 GHz i7

    """

    r, c = dataframe.shape

    idx = [np.random.choice(dataframe.index, size = len(dataframe), replace = replace) for n in range(0, n_shuffles)]

    combos = list(itertools.combinations(dataframe.columns, r = 2))
    combo_labels = [label[0] + '-' + label[1] for label in combos]
    combo_len = len(combos)

    obs_corrs = np.array([method(dataframe[labels[0]].values, dataframe[labels[1]].values)[0] for labels in combos]) #generate unpermuted data

    perm_corrs = np.array(joblib.Parallel(n_jobs = n_jobs, verbose = verbose)(joblib.delayed(perm_corr_single)(dataframe, combos, perm_idx, method = scipy.stats.spearmanr) for perm_idx in idx))

    p_raw = p_perm(obs_corrs, perm_corrs)

    return(obs_corrs, perm_corrs, p_raw, p_maxT, combos)


def plot_corr_perms(dataframe, corr_obs, p, alpha = 0.05):
    '''
    Used to plot distributions of bootstrapped correlation output for troubleshooting + validation

    dataframe -Data (r * c) to be correlated in the form of a pandas dataframe
    corr_obs - Observed correlation values
    p - p values (either raw or corrected). Note: Get output from corr_perm
    alpha - The threshold which a correlation is deemed "significantly" unlikely to be equal to the null.
    '''
    r, c = dataframe.shape
    cols = dataframe.columns

    z = np.zeros([c, c])

    iu = np.triu_indices(c, k = 1)
    il = iu[::-1] #reverse for lower

    sig_idxs = np.ravel(np.where(p < 0.05))

    corr_obs_sig = np.array([corr_obs[n] if n in sig_idxs else 0 for n in range(0, len(corr_obs))])

    z[iu] = corr_obs
    z[il] = corr_obs_sig

    df = pd.DataFrame(data = z, columns = cols, index = cols)

    sns.heatmap(df, vmin = -1, vmax = 1, cmap = 'RdYlBu_r')
