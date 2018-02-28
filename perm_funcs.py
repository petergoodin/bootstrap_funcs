def anova_perm(formula, dataframe, n_shuffles):
    '''
    Uses Manly's method of unrestrained reshuffling across levels to calculate null distribution & hypothesis tests
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
    
    #Make exact number of shuffles (coz python)
    n_shuffles = n_shuffles + 1
    
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
    
    #Loop selecting random indicies, running lmm and collecting test statistic
    for n in range(0, n_shuffles):
        rand_idxs = np.random.uniform(low = 0, high = r, size = r)
        df_shuffle = dataframe.iloc[rand_idxs]
        
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            perm_model = model = smf.ols(formula, data = df_shuffle).fit(reml = False)
        perm_model_f = sm.stats.anova_lm(perm_model, typ = 2)
        perm_f_shuffle = perm_model_f[perm_model_f.index != 'Residual']['F']
        
        for term in terms:
            perm_f[term][n] = perm_f_shuffle.loc[term]

    
    p = {term: sum(perm_f[term] >= obs_f[term]) / n_shuffles for term in terms}
        
    return(obs_f, perm_f, p)


def lmm_perm(formula, group, dataframe, n_shuffles):
    '''
    Uses Manly's method of unrestrained reshuffling across levels to calculate null distribution & hypothesis tests
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
    
    #Make exact number of shuffles (coz python)
    n_shuffles = n_shuffles + 1
    
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
        rand_idxs = np.random.uniform(low = 0, high = r, size = r)
        df_shuffle = dataframe.iloc[rand_idxs]
        
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            perm_model = smf.mixedlm(formula, groups = group, data = df_shuffle).fit(reml = False)
        perm_df_wald = perm_model.wald_test_terms().table
        
        for term in terms:
            perm_wald[term][n] = perm_df_wald['statistic'].loc[term]

    
    p = {term: sum(perm_wald[term] >= obs_wald[term]) / n_shuffles for term in terms}
        
    return(obs_wald, perm_wald, p)


def perm_plot(obs, perm):
    fig, axes = plt.subplots(len(perm.keys()), 1)
    for n, term in enumerate(perm.keys()):       
            sns.distplot(perm[term], ax = axes[n])
            
            #Make perrty
            axes[n].axvline(obs[term], 0, 1, linestyle = '--', color = [1, 0, 0])
            axes[n].set_title(term)
            plt.tight_layout()
            
    plt.show()
