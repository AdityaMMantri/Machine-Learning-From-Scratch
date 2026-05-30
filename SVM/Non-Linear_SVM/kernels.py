import numpy as np

def linear_kernel(x1,x2):
    # K(x, z) = x^T*z
    # Equivalent to normal linear SVM.
    return np.dot(x1,x2)

def polynomial_kernel(x1,x2,degree=2,c=1):
    """
    K(x, z) = (x^T*z + c)^degree

    Captures:
    - feature interactions
    - polynomial relationships
    - curved boundaries
    """
    return np.dot((x1,x2)+c)**degree

def rbf_kernel(x1,x2,gamma=0.5):
    """
    K(x, z) = exp(-gamma*||x-z||^2)

    Measures local similarity.

    Nearby points -> value near 1
    Far points    -> value near 0
    """
    dist_square=np.linalg.norm(x1-x2)**2
    return np.exp(-gamma*dist_square)

def laplacian_kernel(x1,x2,gamma=0.5):
    '''
    K(x, z) = exp(-gamma*||x-z||_1)

    Similar to RBF kernel,
    but uses Manhattan distance (L1 norm)
    instead of Euclidean distance.

    Measures local similarity.

    Nearby points -> value near 1
    Far points    -> value near 0

    Often more robust to:
    - outliers
    - sparse data
    - high-dimensional noise
    '''
    distance=np.linalg.norm(x1-x2,ord=1)
    return np.exp(-gamma*distance)

