def eigen(train_x):
    import numpy as np
    cov_train = np.cov(train_x.T)
    eig_val, eig_vec = np.linalg.eig(cov_train)
    return eig_val, eig_vec

def PCA(train_x, n, eig_val_train, eig_vec_train):
    import numpy as np
    """
    train_x: (ndarray)
    n: n principal components
    """
    eig_pairs = [(np.abs(eig_val_train[i]), eig_vec_train[:,i]) for i in range(len(eig_val_train))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    tot = sum(eig_val_train[:n])
    var_exp = [(i / tot)*100 for i in sorted(eig_val_train[:n], reverse=True)]

    cum_var_exp = np.array(np.cumsum(var_exp), dtype = 'float32')

    pca_w = np.array([eig_pairs[i][1] for i in range(n)], dtype = 'float32')
    return train_x @ pca_w.T