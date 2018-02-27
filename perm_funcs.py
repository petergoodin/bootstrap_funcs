def lmm_perm(formula, group, dataframe, n_shuffles):
    '''
    Uses Manly's method of unrestrained reshuffling across levels to calculate null distribution & hypothesis tests
    % null > obs
    
    NOTE: Tests for main effects / interactions similar to ANOVA (in fact, test statistic is taken from Wald test)
    
    Input:
    formula - Patsy compliant formula for the model
    group - Group variable to use
    dataframe - Pandas dataframe in long form with the data to be modelled.
    n_samples - number of reshuffles to use
    seed - Seed for reproducability
    
    Output:
    p - p value (% null > obs)
    shuffles - Dictionary of  r terms x c shuffles
    '''
    
    #Make exact number of shuffles (coz python)
    n_shuffles = n_shuffles + 1
    
    #Get data info
    r, c = dataframe.shape
    
    # Calculate observed linear mixed model
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        obs_model = model = smf.mixedlm(formula, groups = group, data = dataframe).fit(reml = True)
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
            model = smf.mixedlm(formula, groups = group, data = df_shuffle).fit(reml = True)
        df_wald = model.wald_test_terms().table
        
        for term in terms:
            perm_wald[term][n] = df_wald['statistic'].loc[term]

    
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
