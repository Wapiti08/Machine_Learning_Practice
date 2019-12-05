def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset
    
    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the 
        mean and standard deviation respectively.
    
    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those 
        dimensions when doing normalization.
    """
    mu = np.zeros(X.shape[1]) # <-- EDIT THIS, compute the mean of X
    mu = np.mean(X,axis=0)
    std = np.std(X, axis=0)
    # Return an array copy of the given object
    std_filled = std.copy()
    # NaN will be solved by that
    # std_filled[std==0] = 1.0 (the index with 0 will be reassigned)
    std_filled[std==0] = 1.
    Xbar = (X-mu)/std_filled                 # <-- EDIT THIS, compute the normalized data Xbar
    return Xbar, mu, std

def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors 
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix
    
    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    #use linalg.eig function to compute the eigvals and eigvecs
    eigvals, eigvecs = np.linalg.eig(S)
    # return the indices that would sort an array
    desc_order = np.argsort(eigvals)[::-1]
    sorted_eigvals = eigvals[desc_order]
    sorted_eigvecs = eigvecs[:,desc_order]
    return (sorted_eigvals, sorted_eigvecs) # <-- EDIT THIS to return the eigenvalues and corresponding eigenvectors

def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    
    BT = np.transpose(B)
    #If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred
    # projection_matrix: 投影矩阵 ---- B@(B.T@B)^-1@B.T
    #P = B @ np.linalg.inv(BT @ B) @ BT
    P = B @ np.linalg.pinv(B)
    return P # <-- EDIT THIS to compute the projection matrix

def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    # your solution should take advantage of the functions you have implemented above.
     # Estimate a covariance matrix, given data and weights
    #rowvar---False:each column represents a variable, while the rows contain observations
    # If bias is True, then normalization is by N
    S = np.cov(X, rowvar=False, bias=True)

    # Next find eigenvalues and corresponding eigenvectors for S by implementing eig().
    eig_vals, eig_vecs = eig(S)
    
    # D is the dimensions
    P = projection_matrix(eig_vecs[:,0:num_components]) # projection matrix
    # B @ np.linalg.inv(BT @ B) @ BT @ X.T
    X_reconstruct = (P @ X.T).T
    return X_reconstruct # <-- EDIT THIS to return the reconstruction of X

### PCA for high dimensional datasets

def PCA_high_dim(X, n_components):
    """Compute PCA for small sample size but high-dimensional features. 
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        n_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    
    N, D = X.shape

    M = np.zeros((N, N)) # EDIT THIS, compute the matrix \frac{1}{N}XX^T.

    M = np.cov(X)
    # core part to compute the high dimensions PCA
    M = (1/N) * X @ X.T

    # Next find eigenvalues and corresponding eigenvectors for S by implementing eig().

    eig_vals, eig_vecs = eig(M) # EDIT THIS, compute the eigenvalues.

    eig_vecs = eig_vecs[:,:n_components]

    U = np.transpose(X)@eig_vecs #Compute the eigenvectors for the original PCA problem.

    nu=np.linalg.norm(U,axis=0)#,keepdims=True)

    U=U/nu
    
    return U # <-- EDIT THIS to return the reconstruction of X
