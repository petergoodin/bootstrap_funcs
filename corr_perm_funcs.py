def corr_mcc(dataframe, method = scipy.stats.spearmanr, n_shuffles = 10000, seed = 1010):
    
    """
    
    TO DO: 
    
    Run in parallel.
    
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

    r, c = dataframe.shape

    corr_labels = [l1 + '-' + l2 for l2 in dataframe.columns for l1 in dataframe.columns]

    t = n_shuffles * 0.25

    obs_corr = method(dataframe)[0]

    perm_corr = np.zeros([n_shuffles, c, c])

    for n in range(0, n_shuffles):
        if n % t == 0:
            print((n / n_shuffles) * 100)

        for n_outer, col_outer in enumerate(dataframe.columns):
            x = dataframe[col_outer]
            for n_inner, col_inner in enumerate(dataframe.columns):
                rand_idxs = np.random.choice(dataframe.index, size = len(dataframe), replace = False)
                y_perm = dataframe[col_inner].loc[rand_idxs]
                perm_corr[n, n_outer, n_inner] = method(x, y_perm)[0]


    perm_array = perm_corr.reshape(n_shuffles, c * c)

    obs_array = obs_corr.reshape(c * c)
    maxT = np.zeros(n_shuffles)


    for n in range(0, n_shuffles):
        maxT[n] = np.max(perm_array[n])

    p_ph_array = np.zeros(c*c)


    for n in range(0, n_shuffles):
        idx = np.where(np.max(np.abs(perm_array[n])))
        maxT[n] = perm_array[n, idx]

    for n in range(0, c * c):
        p_ph_array[n] = sum(abs(maxT) > abs(obs_array[n])) / n_shuffles

    p_ph = p_ph_array.reshape([c, c])
    
    print('Complete')
    return(obs_corr, perm_corr, maxT, p_ph)
    
def plot_corr_perms(corr_obs, corr_perms, alpha = 0.05, labels = None):
    '''
    Used to plot distributions of bootstrapped correlation output for troubleshooting + validation
    
    corr_obs - Observed correlation values
    corr_perm - Permutated null correlation values
    alpha - The threshold which a correlation is deemed "significantly" unlikely to be equal to the null.
    labels = Input a list of labels for the correlations
    '''
    
    fig = plt.figure(figsize = [20, 20])
    n_samples, r, c = corr_size = corr_perms.shape

    #Reshape matrices to 1d
    corr_obs_rs = np.reshape(corr_obs, [r * c])
    corr_perms_rs = np.reshape(corr_perms, [n_samples, r * c])
    
    #Create boolean array to test if reshaped value is from matrix diagonal.
    di = np.diag_indices(r)
    zero_mat = np.zeros_like(corr_obs)
    zero_mat[di] = 1
    diag_array = np.reshape(zero_mat, [r * c])
    
    #Create boolean array to store values below alpha
    p_mat = np.zeros_like(corr_obs)
    p_array = np.reshape(p_mat, [r * c])
    

    for n in range(r * c):
        corr_obs_val = corr_obs_rs[n]
        ax = plt.subplot(r, c, n + 1)
        perm_plot_data = corr_perms_rs[:, n]
        sns.distplot(perm_plot_data, hist = False, kde = True, norm_hist = False)

        #Plot obs val on null dist
        obs_plot_data = corr_obs_rs[n]
        xmin, xmax = [ax.dataLim.min[0], ax.dataLim.max[0]]
        ymin, ymax = [ax.dataLim.min[1], ax.dataLim.max[1]]
        plt.grid('on')
        plt.vlines(obs_plot_data, ymin, ymax, colors = [1, 0, 0], linestyles = '--')
        
        p = sum(abs(corr_perms_rs[:, n]) > abs(corr_obs_rs[n])) / n_samples
        plt.xlabel(p)
        
        if p < alpha:
            if corr_obs_rs[n] > 0:
                plt.ylabel('***', fontsize = 25, color = [1,0,0])
            elif corr_obs_rs[n] < 0:
                plt.ylabel('***', fontsize = 25, color = [0,0,1])
            p_array[n] = 1
            

        if labels != None:
            plt.title(labels[n], fontsize = 8)
        
        #Clear diagonal 
        if diag_array[n] == 1: 
            plt.cla()
            plt.axis('off')
        ax.set_facecolor('w')
    plt.tight_layout()
    plt.show()
    
    p_mat = np.reshape(p_array, [r, c])
    
    return(fig, p_mat)

def jitter(vals):
    stdev = .01 * (max(vals) - min(vals))
    return(vals + np.random.randn(len(vals)) * stdev)
