import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class HardMarginLinearSVM:
    def __init__(self,tolerance:float,eps:float,max_passes:int)->None:
        self.tol=tolerance # used for KKT conditions floating point
        self.eps=eps       # used for identifying meangiful updates
        self.max_passes=max_passes # how many times is it allowed to iteratre over data set with out updating
        self.bias=0.0
        self.w=None

    def fit(self,X,y):
        X=np.asarray(X,dtype=float)
        y=np.asarray(y,dtype=float)

        if not set(np.unique(y)).issubset({-1.0,1.0}):
            raise ValueError("Binary SVM requires class labels to be +1 and -1")
        n_rows,n_features=X.shape

        alpha=np.zeros(n_rows)
        self.w = np.zeros(n_features)
        self.bias=0.0
        passes=0

        while passes<self.max_passes:
            num_changed=0
            f= X @ self.w + self.bias  # Error term used in SMO: E_i=f(x_i)-y_i
            E=f-y
            for i in range(n_rows):
                r_i=y[i]*f[i]
                # Case 1: alpha_i == 0→point must be outside margin (r_i >= 1)
                # Case 2: alpha_i > 0→point must lie exactly on margin (r_i == 1)
                violation_conditions=((alpha[i]<self.eps and r_i< 1-self.tol) or (alpha[i]>self.eps and abs(r_i-1)>self.tol)) # hard margin conditons
                if not violation_conditions:
                    continue
                 # by heuristic approach we are maximizing |E_i - E_j|
                diff=np.abs(E[i] - E)
                diff[i]=-np.inf
                j=np.argmax(diff)
                if i==j:
                    continue
                
                eta=np.dot(X[i]-X[j],X[i]-X[j]) #curvature eta = ||x_i - x_j||^2 which is dot product of Xi-Xj*Xi-Xj
                if eta<=0.0:
                    continue

                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                # Lower Bound
                if y[i]!=y[j]:
                    L=max(0.0,alpha_j_old-alpha_i_old)
                else:
                    L=max(0.0,alpha_i_old+alpha_j_old)
                
                # updating alpha_j
                alpha_j_new=alpha_j_old+y[j]*(E[i]-E[j])/eta
                alpha_j_new=max(L,alpha_j_new)

                # condition of minimal update
                if abs(alpha_j_new-alpha_j_old)<self.eps:
                    continue
                alpha_i_new= alpha_i_old-y[i]*y[j]*(alpha_j_new-alpha_j_old) #Updating alpha_i USING EQUALITY CONSTRAINT

                alpha[i]=alpha_i_new
                alpha[j]=alpha_j_new
                    #deriving primal factors
                self.w+=(alpha_i_new-alpha_i_old)*y[i]*X[i]
                self.w+=(alpha_j_new-alpha_j_old)*y[j]*X[j]
                b1=self.bias -E[i]-(alpha_i_new-alpha_i_old)*y[i]*np.dot(X[i],X[i])-(alpha_j_new - alpha_j_old)*y[j]*np.dot(X[i],X[j])
                b2=self.bias -E[j]-(alpha_i_new-alpha_i_old)*y[i]*np.dot(X[i],X[j])-(alpha_j_new - alpha_j_old)*y[j]*np.dot(X[j],X[j])
                
                if alpha_i_new>0.0:
                    self.bias=b1
                elif alpha_j_new>0.0:
                    self.bias=b2
                else:
                    self.bias=0.5*(b1+b2)
                num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

    def predict(self, X):
        X = np.asarray(X)
        return np.sign(X @ self.w + self.bias)



np.random.seed(42)
# Class -1
X_neg = np.random.randn(20, 2) + np.array([2, 2])
# Class +1
X_pos = np.random.randn(20, 2) + np.array([6, 6])

X = np.vstack((X_neg, X_pos))
y = np.hstack((-np.ones(20), np.ones(20)))

SVM_custom = HardMarginLinearSVM(tolerance=1e-3,eps=1e-8,max_passes=5)
SVM_custom.fit(X,y)
scale=np.linalg.norm(SVM_custom.w)
SVM_custom.w /= scale
SVM_custom.bias /= scale
print("Custom SVM w:", SVM_custom.w)
print("Custom SVM b:", SVM_custom.bias)
print("Custom predictions:", SVM_custom.predict(X))


SVM_sckit = SVC(kernel="linear",C=1e6) # C has to be set but very low turns to hard margin
SVM_sckit.fit(X, y)

w_sk = SVM_sckit.coef_[0]
b_sk = SVM_sckit.intercept_[0]

print("Sklearn SVM w:", w_sk)
print("Sklearn SVM b:", b_sk)
print("Sklearn predictions:", SVM_sckit.predict(X))

plt.figure(figsize=(8, 6))
plt.scatter(
    X[y == -1][:, 0],
    X[y == -1][:, 1],
    color="red",
    label="Class -1"
)
plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    color="blue",
    label="Class +1"
)
x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

y_custom = -(SVM_custom.w[0] * x_vals + SVM_custom.bias) / SVM_custom.w[1]
plt.plot(
    x_vals,
    y_custom,
    "k--",
    linewidth=2,
    label="Custom SMO (Hard Margin)"
)

y_sklearn = -(w_sk[0] * x_vals + b_sk) / w_sk[1]
plt.plot(
    x_vals,
    y_sklearn,
    "g-",
    linewidth=2,
    label="Sklearn SVM (C → ∞)"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Hard-Margin SVM: Custom SMO vs Scikit-learn")
plt.legend()
plt.grid(True)
plt.show()



