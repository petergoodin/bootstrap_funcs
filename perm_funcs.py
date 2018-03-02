def t_perm(m1, m2, n_shuffles, term, method = scipy.stats.ttest_rel):
    
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

    p = {}    

    p[term] = sum(abs(perm_t[term]) >= abs(obs_t[term])) / n_shuffles
    
    return(obs_t, perm_t, p)



def post_hoc_perm(conditions, n_shuffles, dataframe, method = scipy.stats.ttest_rel):
    
    """
    Calculates post-hoc analses using the maxT correction from multiple comparisons.
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
    
    pairs = [pair for pair in itertools.combinations(conditions, 2)]
    n_pairs = len(pairs)

    obs_cond = {}
    perm_cond = {}
    p_cond = {}


    for pair in pairs:
        term = pair[0] + '_vs_' + pair[1]
        obs, perm, p = t_perm(dataframe[pair[0]], dataframe[pair[1]], n_shuffles, term)
        obs_cond.update(obs)
        perm_cond.update(perm)
        p_cond.update(p)

    maxT = {key: np.max(perm_cond[key]) for key in perm_cond.keys()}

    p_ph = {term: sum(maxT.values() >= abs(obs_cond[term])) / n_pairs for term in maxT.keys()}
    
    return(perm_cond, maxT, p_ph)
        
        
def anova_perm(formula, shuffle_var, dataframe, n_shuffles):
    '''
    Uses Manly's method (2007) of unrestrained reshuffling across levels to calculate null distribution & hypothesis tests
    % shuffles > obs
    
    Input:
    formula - Patsy compliant formula for the model
    dataframe - Pandas dataframe with the data to be modelled.
    n_shuffles - number of reshuffles to use
    seed - Seed for reproducability
    
    Output:
    obs_f - Observed statistic from the f test
    perm_f - Permutated statistic from the f test
    p - % shuffles > obs 
    '''

    t = np.round(n_shuffles / 4, decimals = 0)
    
    #Get data info
    r, c = dataframe.shape
    
    # Calculate observed linear mixed model
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        obs_model = model = smf.ols(formula, data = dataframe).fit(reml = False)
    obs_model_f = sm.stats.anova_lm(obs_model, typ = 2)
    obs_f = obs_model_f[obs_model_f.index != 'Residual']['F']
    
    #Capture information from output wald results (minus intercept)
    terms = obs_f.index.values
    
    #Preallocate memory
    perm_f = {term: np.zeros(n_shuffles) for term in terms}    
    
    #Loop selecting random indicies, running ANOVA and collecting test statistic
    for n in range(0, n_shuffles):
        if n % t == 0:
            print((n / n_shuffles) * 100)
            
        shuffle_vals = np.random.choice(dataframe[shuffle_var], size = len(dataframe), replace = False)
        df_shuffle = dataframe.copy()
        df_shuffle[shuffle_var] = shuffle_vals
        
        perm_model = smf.ols(formula, data = df_shuffle).fit(reml = False)
        perm_model_f = sm.stats.anova_lm(perm_model, typ = 2)
        perm_f_shuffle = perm_model_f[perm_model_f.index != 'Residual']['F']
        
        for term in terms:
            perm_f[term][n] = perm_f_shuffle.loc[term]

    
    p = {term: sum(perm_f[term] >= obs_f[term]) / n_shuffles for term in terms}
    print('Complete')
    return(obs_f, perm_f, p)



def lmm_perm(formula, shuffle_var, group, dataframe, n_shuffles):
    '''
    Uses Manly's method (2007) of unrestrained reshuffling across levels to calculate null distribution & hypothesis tests
    % shuffles > obs
    
    NOTE: Tests for main effects / interactions similar to ANOVA (in fact, null test statistic is the Wald test)
    
    Input:
    formula - Patsy compliant formula for the model
    group - Group variable to use
    dataframe - Pandas dataframe with the data to be modelled.
    n_samples - number of reshuffles to use
    seed - Seed for reproducability
    
    Output:
    obs_wald - Observed statistic from the Wald test
    perm_wald - Permutated statistic from the Wald test
    p - % shuffles > obs 
    '''

    #Percent finished
    t = np.round(n_shuffles / 4, decimals = 0)
    
    #Get data info
    r, c = dataframe.shape
    
    # Calculate observed linear mixed model
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("once")
        obs_model = model = smf.mixedlm(formula, groups = group, data = dataframe).fit(reml = False)
    obs_wald_table = model.wald_test_terms().table
    obs_wald = obs_wald_table[obs_wald_table.index != 'Intercept']['statistic']
    
    #Capture information from output wald results (minus intercept)
    terms = obs_wald.index.values
    
    #Preallocate memory
    perm_wald = {term: np.zeros(n_shuffles) for term in terms}    
    
    #Loop selecting random indicies, running lmm and collecting test statistic
    for n in range(0, n_shuffles):
        if n % t == 0:
            print((n / n_shuffles) * 100)
            
        shuffle_vals = np.random.choice(dataframe[shuffle_var], size = len(dataframe), replace = False)
        df_shuffle = dataframe.copy()
        df_shuffle[shuffle_var] = shuffle_vals
        
        
        with warnings.catch_warnings():
            warnings.filterwarnings("once")
            perm_model = smf.mixedlm(formula, groups = group, data = df_shuffle).fit(reml = False)
        perm_df_wald = perm_model.wald_test_terms().table
        
        for term in terms:
            perm_wald[term][n] = perm_df_wald['statistic'].loc[term]

    
    p = {term: sum(perm_wald[term] >= obs_wald[term]) / n_shuffles for term in terms}
    
    print('Completed')  
    return(obs_wald, perm_wald, p)


def perm_plot(obs, perm, p, fig_title):
    
    """
    Plots combined histogram / KDE from permutation plot using seaborn.
    Adds vertical lines for observed and threshold
    
    
    """
    plot_rows = len(perm.keys())
    
    fig, axes = plt.subplots(plot_rows, 1)

    for n, term in enumerate(perm.keys()):

        if plot_rows > 1:
            sns.distplot(perm[term], ax = axes[n])
            thresh = np.percentile(perm[term], 95, interpolation = 'nearest')

            #Formatting
            axes[n].axvline(obs[term], 0, 1, linestyle = '--', color = [1, 0, 0], label = 'Observed')
            axes[n].axvline(thresh, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
            axes[n].set_title(term, fontsize = 16, x = 0.1, y = 1.05)
            axes[n].set_xlabel('Permuted Test Value', fontsize = 15)
            axes[n].text(0.6, 0.5, 'p = ' + str(np.round(p[term], decimals = 5)), fontsize = 20, transform = axes[n].transAxes)

            for tick in axes[n].xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in axes[n].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            
            if n ==  np.around(plot_rows / 2, decimals = 0) - 1:
                axes[n].legend(fontsize = 20, loc = "center left", bbox_to_anchor = (1, 0.5), numpoints = 1)


        else:
            sns.distplot(perm[term], ax = axes)
            thresh = np.percentile(perm[term], 95, interpolation = 'nearest')

            #Formatting
            axes.axvline(obs[term], 0, 1, linestyle = '--', color = [1, 0, 0], label = 'Observed')
            axes.axvline(thresh, 0, 1, linestyle = '-', color = [0, 0, 0], label = 'Threshold')
            axes.set_title(term, fontsize = 16, x = 0.1, y = 1.05)
            axes.set_xlabel('Permuted Test Value', fontsize = 15)
            axes.text(0.6, 0.5, 'p = ' + str(np.round(p[term], decimals = 2)), fontsize = 20, transform = axes.transAxes)

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
