def anova_perm(formula, shuffle_var, dataframe, n_shuffles):
    '''
    Uses Manly's method (2007) of unrestrained reshuffling across levels to calculate null distribution & hypothesis tests
    % shuffles > obs
    
    Input:
    formula - Patsy compliant formula for the model
    dataframe - Pandas dataframe in long form with the data to be modelled.
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
    
    NOTE: Tests for main effects / interactions similar to ANOVA (in fact, test statistic is taken from Wald test)
    
    Input:
    formula - Patsy compliant formula for the model
    group - Group variable to use
    dataframe - Pandas dataframe in long form with the data to be modelled.
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
        warnings.filterwarnings("ignore")
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
            warnings.filterwarnings("ignore")
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
            axes[n].text(0.6, 0.5, 'p = ' + str(np.round(p[term], decimals = 2)), fontsize = 20, transform = axes[n].transAxes)

            for tick in axes[n].xaxis.get_major_ticks():
                tick.label.set_fontsize(15)
            for tick in axes[n].yaxis.get_major_ticks():
                tick.label.set_fontsize(15)


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

            axes.legend(fontsize = 20)

    if fig_title != None:        
        fig.suptitle(fig_title, fontsize = 24, y = 1.05)         
    
    plt.tight_layout()       
    plt.show()
    
    return(fig, axes)
