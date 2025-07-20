import numpy as np

class PCA:

    def __init__(self,n_components):
        self.n_components=n_components # no of principal component axes
        self.mean=None
        self.explained_variance=None # will store eigen values
        self.components=None # to store eigen vectors

    def fit(self,X,tolerance=1e-6,max_iterations=1000):
        self.mean=np.mean(X,axis=0) # mean of each feature
        X_centered=X-self.mean
        cov_matrix=np.cov(X_centered,rowvar=False)

        eigenvalues, eigenvectors = self.qr_algorithm(cov_matrix,max_iterations)
        sorted_eigen_value_index=np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_eigen_value_index][:self.n_components]
        self.components = eigenvectors[:, sorted_eigen_value_index][:, :self.n_components]
    
    def transform(self, X):
        # Project original data onto principal component axes
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def gram_schmidt(self,A):
        n,m=A.shape # A is the  matrix whose eigenvalues and eigenvectors we want.
        Q=np.zeros((n,m)) # orthonormal matrix
        R=np.zeros((n,m)) # upper triangular matrix
        for j in range(m):
            v=A[:,j] # v is the current feature
            for i in range(j): # this loop is skipped for first iteration
                R[i,j]=np.dot(Q[:,i],A[:,j])
                v=v-R[i,j]*Q[i,j]
            R[j,j]=np.linalg.norm(v)
            if R[j,j]==0:
                Q[:,j]=v
            else:
                Q[:,j]=v/R[j,j]
            
        return Q,R

    def qr_algorithm(self,A,max_iterations):
        A_k=A.copy()
        n=A.shape[0]
        Q_total=np.eye(n) # identity matrix 
        for i in range(max_iterations):
            Q,R=self.gram_schmidt(A_k)
            A_k=R@Q 
            Q_total=Q_total@Q # eigen vector,this is the cumulative product of all Q matrices:
        
        eigenvalues=np.diag(A_k)
        return eigenvalues,Q_total
        
from sklearn.datasets import load_iris
X = load_iris().data

pca = PCA(n_components=4)
pca.fit(X)
X_reduced = pca.transform(X)

print("Explained Variance:", pca.explained_variance)
print("Principal Components:\n", pca.components)
print("Reduced shape:", X_reduced.shape)

