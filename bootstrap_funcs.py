import numpy as np

def bs_mv(x, groups):
    '''
    Evalutes model generalisability via bootstrapping. 
    The method will take n observations across g groups
    and for each iteration create training and testing datasets
    for each group using the list in groups. Training datasets are created 
    using a uniform distribution (with replacement) of x and testing datasets
    are created using those observations not used in the training set.     

    estimator = Instance of estimation method
    x = Array or matrix of values used for modelling
    groups = List or array of which rows are associated with which group
    ints = Number of iterations to bootstrap
    '''

    n_groups, n_group_counts = np.unique(groups, return_counts = True) #Return unique values of groups and counts of values
    scores = [] #Empty list to keep the output of each bootstrapped estimation


    for n, group in enumerate(n_groups):

        if n == 0:
            group_idx = np.squeeze(np.where(groups == group))
        
            train_idx = np.random.choice(group_idx, size = n_group_counts[n]) #Select random observations with replacement of size group using uniform distribution
            x_train = x[train_idx]
            y_train = groups[train_idx]

            test_idx = np.random.choice(group_idx[np.in1d(group_idx, train_idx, invert = True)], size = n_group_counts[n]) #Use observations not included in training data
            x_test = x[test_idx]
            y_test = groups[test_idx]

        else:
            group_idx = np.squeeze(np.where(groups == group))
        
            train_idx = np.random.choice(group_idx, size = n_group_counts[n]) #Select random observations with replacement of size group using uniform distribution
            x_train = np.vstack([x_train, x[train_idx]])
            y_train = np.hstack([y_train, groups[train_idx]])

            test_idx = np.random.choice(group_idx[np.in1d(group_idx, train_idx, invert = True)], size = n_group_counts[n]) #Use observations not included in training data
            x_test = np.vstack([x_test, x[test_idx]])
            y_test = np.hstack([y_test, groups[test_idx]])

    return(x_train, y_train, x_test, y_test)

        

    


    
    

    

