import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers
from sklearn.datasets import make_blobs

class HardMarginLinearSVM:
    def fit(self,X,y):
        X=np.asarray(X,dtype=float) # [n_rows*n_features]
        y=np.asarray(y,dtype=float) # 1D array
        unique_labels=np.unique(y)
        if set(unique_labels)=={0,1}:
            y=np.where(y==0,-1,1)
        elif set(unique_labels) == {-1, 1}:
            pass
        else:
            raise ValueError("Labels must be {0,1} or {-1,+1}")
        n_rows,n_features=X.shape

        #Linear kernel matrix
        K=X@X.T # [n*n]
        P=matrix(np.outer(y,y)*K,tc='d') # simple hai yi*yj karna hai matalb rowi*columnj 
        q=matrix(-np.ones(n_rows),tc='d') # [n*1]
        G=matrix(-np.eye(n_rows),tc='d') # [n*n]
        h=matrix(np.zeros(n_rows),tc='d') # [n*n]
        A=matrix(y.reshape(1,-1),tc='d') # 1 means 1 row and -1 means infinite amounts of columns so [1*n]
        b=matrix(0.0) # [1*1] ie... a scalar number

        solution=solvers.qp(P,q,G,h,A,b)
        alpha=np.array(solution["x"]).flatten()
        self.w = np.sum((alpha * y)[:, None] * X, axis=0)

        support_vector = alpha>1e-6
        self.b = np.mean(y[support_vector]-X[support_vector]@ self.w)
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X @ self.w + self.b)
    
def plot_svm(X, y, svm):
    plt.figure(figsize=(6,6))
    plt.scatter(X[y==1][:,0], X[y==1][:,1], label="+1")
    plt.scatter(X[y==-1][:,0], X[y==-1][:,1], label="-1")

    x_vals = np.linspace(-5, 5, 100)
    decision = -(svm.w[0]*x_vals + svm.b) / svm.w[1]
    margin1 = -(svm.w[0]*x_vals + svm.b - 1) / svm.w[1]
    margin2 = -(svm.w[0]*x_vals + svm.b + 1) / svm.w[1]

    plt.plot(x_vals, decision, 'k-')
    plt.plot(x_vals, margin1, 'k--')
    plt.plot(x_vals, margin2, 'k--')

    plt.legend()
    plt.grid(True)
    plt.title("Hard-Margin Linear SVM (using external QP solver)")
    plt.show()


np.random.seed(0)
X_pos=np.random.randn(20, 2) + np.array([2, 2])
y_pos=np.ones(20)

X_neg=np.random.randn(20, 2) + np.array([-2, -2])
y_neg=-np.ones(20)

X=np.vstack((X_pos, X_neg))
y=np.hstack((y_pos, y_neg))

svm=HardMarginLinearSVM()
svm.fit(X, y)

preds=svm.predict(X)
print("Training accuracy:", np.mean(preds == y))
plot_svm(X, y, svm)
