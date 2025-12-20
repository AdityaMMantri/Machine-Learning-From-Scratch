import numpy as np
import matplotlib.pyplot as plt

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
    plt.title("Hard-Margin Linear SVM (Dual Gradient Approximation)")
    plt.show()

class HardMarginLinearSVM:
    def __init__(self,learning_rate,iterations):
        self.lr=learning_rate
        self.iters=iterations

    def fit(self,X,y):
        X=np.asarray(X,dtype=float)
        y=np.asarray(y)
        unique_labels=np.unique(y) # to get the different labels present in the traget column
        assert len(unique_labels)==2, "Binary Hard Margin SVM requires two classes +1 and -1" # if the lenght of the unique array will not be 2 the program will stop executing

        K=X @ X.T # linear kernel 
        n_rows, features=X.shape
        alpha= np.zeros(n_rows) # each row will have it's own contraint and hence will get it's own multiplier

        for iter in range(self.iters):
            gradient=np.ones(n_rows) - y * (K @ (alpha * y)) # differentiate the lagranian with respect to alpha
            alpha+=self.lr*gradient

            #KKT conditions
            # all alpha should be positive and only support vectors should have non zero alpha 
            alpha=np.maximum(alpha,0)
            # for condition Σ α_i y_i = 0
            correction=np.dot(alpha, y) / np.dot(y, y)
            alpha-=correction * y

        self.alpha = alpha

        # primal parameters w and b
        self.w = np.sum((alpha * y)[:, None] * X, axis=0)

        support_vector = alpha>1e-6
        # b = y-w*x
        self.b = np.mean(y[support_vector]-X[support_vector]@ self.w)
        margins = y * (X @ self.w + self.b)
        min_margin = np.min(margins)

        self.w = self.w / min_margin
        self.b = self.b / min_margin

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X @ self.w + self.b)

np.random.seed(0)
X_pos=np.random.randn(20, 2) + np.array([2, 2])
y_pos=np.ones(20)

X_neg=np.random.randn(20, 2) + np.array([-2, -2])
y_neg=-np.ones(20)

X=np.vstack((X_pos, X_neg))
y=np.hstack((y_pos, y_neg))

svm=HardMarginLinearSVM(learning_rate=0.01, iterations=5000)
svm.fit(X, y)

preds=svm.predict(X)
print("Training accuracy:", np.mean(preds == y))
plot_svm(X, y, svm)
