from __future__ import division

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pylab as plt
import scipy.stats as stats
import pyvttbl as pt
import itertools
import time
import multiprocessing as mp

def t_perm(m1, m2, n_shuffles, term, method = scipy.stats.ttest_rel, seed = 1010):
    
    np.random.seed(seed)
    
    m1_size = len(m1)
    m2_size = len(m2)

    combined_measures = np.hstack([m1, m2])

    #Observed t value
    obs_t = {term: method(m1, m2)[0]}

    #Storage dictionary
    perm_t = {term: np.zeros(n_shuffles)}
    
    for n in range(0, n_shuffles):

        shuffle_combine_measures = np.random.choice(combined_measures, size = len(combined_measures), replace = False)

        m1_shuffle = shuffle_combine_measures[:m1_size]
        m2_shuffle = shuffle_combine_measures[m2_size:]

        perm_t[term][n] = method(m1_shuffle, m2_shuffle)[0]

    p_t = {}    

    p_t[term] = sum(abs(perm_t[term]) >= abs(obs_t[term])) / n_shuffles
    
    return(obs_t, perm_t, p_t)


def post_hoc_perm(conditions, n_shuffles, dataframe, method = scipy.stats.ttest_rel, seed = 1010):
    
    """
    Calculates post-hoc analses using the maxT correction from multiple comparisons (see Ge, Dudoit & Speed, 2003).
    maxT generates a meta distribution based on the maximum observed statistic from a series
    post-hoc comparisons.
    
    Each obs is then compared against the null maxT distribution using the obs > null / n_shuffles method
    
    Input:
    conditions - A list of the conditions to generate post-hoc analysis on
    n_shuffles - The number of re-shuffles to generate the null distribution
    method - The test statistic to use (default is dependent samples t test).
    NOTE: Calls t_perm internally. Method is used by t_perm which takes two groups / measures.
    Any more groups / measures will require hacking.
    
    Output:
    perm_cond - Permutations for each condition
    maxT - maxT for each condition
    p_ph - Post-hoc tests on whether to reject / keep null
    
    """
    
    np.random.seed(seed)

    pairs = [pair for pair in itertools.combinations(conditions, 2)]
    n_pairs = len(pairs)

    t = np.floor(n_pairs * 0.25)

    obs_cond = {}
    perm_cond = {}
    p_cond = {}
    p_ph = {}

    maxT = np.zeros(n_shuffles)

    #First loop: Generate permutations
    for n, pair in enumerate(pairs):

        if n % t == 0:
            print((n / n_pairs) * 100)

        term = pair[0] + '_vs_' + pair[1]
        obs, perm, p = t_perm(dataframe[pair[0]], dataframe[pair[1]], n_shuffles, term)
        obs_cond.update(obs)
        perm_cond.update(perm)
        p_cond.update(p)



    for n in range(0, n_shuffles):
        shuffle = np.array([shuffles[n] for shuffles in perm_cond.values()])
        maxT[n] = shuffle[np.squeeze(np.where(abs(shuffle) == np.max(np.abs(shuffle))))]

    p_ph = {cond: sum(abs(maxT) >= abs(obs_cond[cond])) / n_shuffles for cond in obs_cond.keys()}
    
    print('Complete')
    return(obs_cond, perm_cond, maxT, p_ph)



def anova_perm(dv, dataframe, n_shuffles, seed = 1010, wfactors = None, bfactors = None, sub = None):
    '''
    Uses pyvttbl to calculate anova and Manly's method (2007) of unrestrained reshuffling across levels to 
    calculate null distribution & hypothesis tests
    % shuffles > obs
    
    Input:
    
    dv - Dependant variable
    wfactors - Within group factors (list)
    bfactors - Between group factors (list)
    dataframe - Pandas dataframe with the data to be modelled.
    n_shuffles - number of reshuffles to use
    seed - Seed for reproducability
    
    Output:
    obs_f - Observed statistic from the f test (dict)
    perm_f - Permutated statistic from the f test (dict)
    p - % shuffles > obs (dict)
    '''
    
    np.random.seed(seed)

    t = np.round(n_shuffles / 4, decimals = 0)

    #Get data info
    r, c = dataframe.shape

    #Number of variables in model
    if wfactors != None and bfactors != None:
        n_vars = len(wfactors + bfactors)
    elif wfactors != None and bfactors == None:
        n_vars = len(wfactors)
    else:
        n_vars = len(bfactors)

    n_effects = np.math.factorial(n_vars)

    #Convert Pandas dataframe to pyvttbl
    pt_df = pt.DataFrame(dataframe)

    aov = pt_df.anova(dv = dv, wfactors = wfactors, bfactors = bfactors, sub = sub)

    if n_vars > 1:
        terms = [aov.keys()[n][0] if len(aov.keys()[n]) < 2 else '*'.join(aov.keys()[n]) for n in range(0, n_effects + 1)]
    else:
        terms = [aov.keys()[n][0] for n in range(0, n_vars)]

    f_obs = {term: aov.values()[n]['F'] for n, term in enumerate(terms)}
    f_perm = {term: [] for term in terms}

    df_shuffle = pt_df.copy()

    for n in range(0, n_shuffles):
        if n % t == 0:
            print((n / n_shuffles) * 100)

        shuffle_vals = np.random.choice(pt_df[dv], size = len(pt_df[dv]), replace = False)
        df_shuffle[dv] = shuffle_vals

        aov_perm = df_shuffle.anova(dv = dv, wfactors = wfactors, bfactors = bfactors, sub = sub)
        [f_perm[term].append(aov_perm.values()[n]['F']) for n, term in enumerate(terms)]

        p = {term: sum(f_perm[term] >= f_obs[term]) / n_shuffles for term in terms}
    print('Complete')
    
    return(f_obs, f_perm, p)



def perm_plot(obs, perm, p, fig_title, tails = 1):
    
    """
    Plots combined histogram / KDE from permutation plot using seaborn.
    Adds vertical lines for observed and threshold.
    
    
    obs = Observed value
    perm = Permutated values
    p = p from permutation test
    tails = Either -1 (left tail), 1 (right tail) or 2 (two tailed)
    
    
    """
    plot_rows = len(perm.keys())
    
    fig, axes = plt.subplots(plot_rows, 1)

    for n, term in enumerate(perm.keys()):

        if plot_rows > 1:
            sns.distplot(perm[term], ax = axes[n], norm_hist = True)

            #Formatting
            axes[n].axvline(obs[term], 0, 1, linestyle = '--', color = [1, 0, 0], label = 'Observed')
            
            if tails == -1:
                thresh = np.percentile(perm[term], 5, interpolation = 'nearest')
                axes[n].axvline(thresh, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
            
            
            if tails == 1:
                thresh = np.percentile(perm[term], 95, interpolation = 'nearest')
                axes[n].axvline(thresh, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
                
            elif tails == 2:
                thresh = np.percentile(perm[term], [2.5, 97.5], interpolation = 'nearest')
                axes[n].axvline(thresh[0], 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
                axes[n].axvline(thresh[1], 0, 1, linestyle = '-', color = [0, 0, 0])
                   
            axes[n].set_title(term, fontsize = 16, x = 0.1, y = 1.05)
            axes[n].set_xlabel('Permuted Test Value', fontsize = 15)
            if p[term] < 0.001:
                axes[n].text(0.6, 0.5, 'p < 0.001', fontsize = 20, transform = axes[n].transAxes)
            else:
                axes[n].text(0.6, 0.5, 'p = ' + str(np.round(p[term], decimals = 5)), fontsize = 20, transform = axes[n].transAxes)    
            

            for tick in axes[n].xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in axes[n].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            
            if n ==  np.around(plot_rows / 2, decimals = 0) - 1:
                axes[n].legend(fontsize = 20, loc = "center left", bbox_to_anchor = (1, 0.5), numpoints = 1)


        else:
            sns.distplot(perm[term], ax = axes, norm_hist = True)

            #Formatting
            axes.axvline(obs[term], 0, 1, linestyle = '--', color = [1, 0, 0], label = 'Observed')
            
            if tails == -1:
                thresh = np.percentile(perm[term], 5, interpolation = 'nearest')
                axes.axvline(thresh, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
            
            
            if tails == 1:
                thresh = np.percentile(perm[term], 95, interpolation = 'nearest')
                axes.axvline(thresh, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
                
            elif tails == 2:
                thresh = np.percentile(perm[term], [2.5, 97.5], interpolation = 'nearest')
                axes.axvline(thresh[0], 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
                axes.axvline(thresh[1], 0, 1, linestyle = '-', color = [0, 0, 0])
            
                       
            axes.set_title(term, fontsize = 16, x = 0.1, y = 1.05)
            axes.set_xlabel('Permuted Test Value', fontsize = 15)
            if p[term] < 0.001:
                axes.text(0.6, 0.5, 'p < 0.001', fontsize = 20, transform = axes.transAxes)
            else:
                axes.text(0.6, 0.5, 'p = ' + str(np.round(p[term], decimals = 5)), fontsize = 20, transform = axes.transAxes)    
            
            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)

            axes.legend(fontsize = 20, loc = "center left", bbox_to_anchor = (1, 0.5), numpoints = 1)

    if fig_title != None:        
        fig.suptitle(fig_title, fontsize = 24, y = 1.05)         
    
    plt.tight_layout()       
    plt.show()
    
    return(fig, axes)


def post_hoc_perm_plot(obs_cond, perm_cond, maxT, fig_title, remove_list = None):
    
    plot_rows = len(perm_cond.keys())

    fig, axes = plt.subplots(plot_rows, 1, figsize = [15, 25])

    for n, term in enumerate(sorted(perm_cond.keys())):
        
        if remove_list != None:
            mod_key = ' '.join([word for word in term.split('_') if word not in remove_list])
        else:
            mod_key = term

        sns.distplot(maxT, ax = axes[n], bins = 5, norm_hist = True)
        thresh_lo, thresh_hi = np.percentile(maxT, [2.5, 97.5])

        #Formatting
        axes[n].axvline(obs_cond[term], 0, 1, linestyle = '--', color = [1, 0, 0], label = 'Observed')
        axes[n].axvline(thresh_lo, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
        axes[n].axvline(thresh_hi, 0, 1, linestyle = '-', color = [0, 0, 0])
        axes[n].set_xlabel(mod_key, fontsize = 25);
        if p_ph[term] < 0.0001:
            axes[n].text(0.7, 0.5, 'p < 0.0001', fontsize = 25, transform = axes[n].transAxes);
        else:
            axes[n].text(0.7, 0.5, 'p = ' + str(np.round(p_ph[term], decimals = 5)), fontsize = 25, transform = axes[n].transAxes)    


        for tick in axes[n].xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
        for tick in axes[n].yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        if n ==  np.around(plot_rows / 2, decimals = 0) - 1:
            axes[n].legend(fontsize = 20, loc = "center left", bbox_to_anchor = (1, 0.5), numpoints = 1)
            
    plt.tight_layout()        
    plt.show()
    
    return(fig, axes)


