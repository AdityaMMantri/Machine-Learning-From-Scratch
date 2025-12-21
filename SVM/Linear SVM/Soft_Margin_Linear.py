import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SoftMarginLinearSVM:
    def __init__(self,C:float,tolerance:float,eps:float,max_passes:int)->None:
        self.C=C # penalty for margin violation, classification error
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
            for i in range(n_rows):
                # KKT conditions require:
                # 1) alpha_i = 0        → point must be outside margin      → r_i >= 1
                # 2) 0 < alpha_i < C    → point must lie exactly on margin  → r_i == 1
                # 3) alpha_i = C        → point inside margin / misclass    → r_i <= 1
                # A violation occurs when alpha_i's value contradicts
                # We allow small numerical tolerance using tol and eps.
                f_i=np.dot(X[i],self.w) + self.bias
                E_i=f_i-y[i]
                violation_conditions=((y[i]*E_i<-self.tol and alpha[i]<self.C)or(y[i]*E_i>self.tol and alpha[i]>0))
                if not violation_conditions:
                    continue
                f=X@self.w+self.bias
                E=f-y
                diff=np.abs(E_i - E)
                diff[i]=-np.inf
                j=np.argmax(diff)
                if i==j:
                    continue
    
                E_j=E[j]
                eta=np.dot(X[i],X[i]) + np.dot(X[j],X[j]) - 2*np.dot(X[i],X[j]) #curvature eta = ||x_i - x_j||^2 which is dot product of Xi-Xj*Xi-Xj
                if eta<=0.0:
                    continue
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                
                # Computing lower annd upper bound to clip the alpha's
                if y[i]!=y[j]:
                    L=max(0.0,alpha_j_old-alpha_i_old)
                    H=min(self.C, self.C+alpha_j_old-alpha_i_old)
                else:
                    L=max(0.0, alpha_i_old+alpha_j_old-self.C)
                    H=min(self.C,alpha_i_old+alpha_j_old)

                if L==H:
                    continue # skip the pair
                alpha_j_new=alpha_j_old+y[j]*(E_i-E_j)/eta
                alpha_j_new=np.clip(alpha_j_new,L,H) # min(max(alpha,L,H))

                # condition of minimal update
                if abs(alpha_j_new-alpha_j_old)<self.eps:
                    continue

                alpha_i_new= alpha_i_old-y[i]*y[j]*(alpha_j_new-alpha_j_old) #Updating alpha_i USING EQUALITY CONSTRAINT

                alpha[i]=alpha_i_new
                alpha[j]=alpha_j_new

                #deriving primal factors
                self.w+=(alpha_i_new-alpha_i_old)*y[i]*X[i]
                self.w+=(alpha_j_new-alpha_j_old)*y[j]*X[j]
                b1=self.bias-E_i- (alpha_i_new-alpha_i_old)*y[i]*np.dot(X[i],X[i])- (alpha_j_new-alpha_j_old)*y[j]*np.dot(X[i],X[j])
                b2=self.bias-E_j- (alpha_i_new-alpha_i_old)*y[i]*np.dot(X[i],X[j])- (alpha_j_new-alpha_j_old)*y[j]*np.dot(X[j],X[j])

                if 0<alpha_i_new<self.C:
                    self.bias=b1
                elif 0<alpha_j_new<self.C:
                    self.bias=b2
                else:
                    self.bias=0.5*(b1+b2)
                num_changed += 1
            if num_changed == 0:
                passes += 1
            else:
                passes = 0

    def predict(self, X):
        X=np.asarray(X, dtype=float)
        return np.where(X @ self.w + self.bias >= 0, 1, -1)

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.2,
    flip_y=0.1,
    random_state=42
)

y = np.where(y == 0, -1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

my_svm=SoftMarginLinearSVM(C=0.65,tolerance=1e-6,eps=1e-14,max_passes=100)
my_svm.fit(X, y)
y_pred_my = my_svm.predict(X)

sk_svm=SVC(kernel="linear",C=1.0)

sk_svm.fit(X, y)
y_pred_sk = sk_svm.predict(X)

# ---------------- ACCURACY COMPARISON ----------------
print("My SMO SVM accuracy:", accuracy_score(y, y_pred_my))
print("sklearn SVM accuracy:", accuracy_score(y, y_pred_sk))

# ---------------- PLOT DECISION BOUNDARIES ----------------
def plot_decision_boundary(w, b, X, y, title):
    plt.figure()
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], marker='o')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x')

    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals)
    plt.title(title)
    plt.show()

plot_decision_boundary(my_svm.w, my_svm.bias, X, y, "My SMO Soft-Margin SVM")
plot_decision_boundary(sk_svm.coef_[0], sk_svm.intercept_[0], X, y, "sklearn Linear SVM")
