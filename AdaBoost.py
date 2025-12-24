import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class DecisionStump:
    def __init__(self):
        self.feature=None
        self.threshold=None
        # Direction of inequality:
        #+1→ predict -1 when x < threshold
        #-1→ predict -1 when x > threshold
        self.polarity=1
        self.alpha=0.0 # Weight/importance for each stump

    def fit(self,X,y,w):
        n_rows,n_features=X.shape
        min_error=float("inf")

        for j in range(n_features):
            values=np.unique(X[:, j])
            if len(values)==1:
                continue
            midpoints=(values[:-1] + values[1:])/2   #Compute midpoints between consecutive values
            thresholds=np.unique(np.concatenate([values, midpoints])) #Use both original values and midpoints as thresholds
            
            for threshold in thresholds:
                for p in [-1,1]:
                    preds=np.ones(n_rows)
                    preds[p*X[:,j]<p*threshold]=-1 #Flip prediction to -1 based on polarity and threshold
                    error=np.sum(w[preds!=y])

                    if error<min_error:
                        min_error=error
                        self.polarity=p
                        self.feature=j
                        self.threshold=threshold
        return min_error
    
    def predict(self,X):
        preds=np.ones(X.shape[0])
        preds[self.polarity*X[:,self.feature]<self.polarity*self.threshold]=-1
        return preds
    
class AdaBoosting:
    def __init__(self,n_estimators):
        self.n_estimators=n_estimators #Maximum number of weak learners (stumps)
        self.stumps: list[DecisionStump] = []
        self.alphas=[]

    def fit(self,X,y):
            if set(np.unique(y))=={0, 1}:
                y=2*y-1
            n_rows,n_features=X.shape
            weights=np.ones(n_rows)/n_rows #Initialize all sample weights uniformly
            eps = 1e-10     #Small constant to avoid division by zero / log(0)

            for i in range(self.n_estimators):
                stump=DecisionStump()
                error=stump.fit(X,y,w=weights)
                error = np.clip(error, 1e-10, 1 - 1e-10)
                alpha=0.5*np.log((1-error)/error) # say/importance of each stump in the final function
                stump.alpha=alpha
                preds=stump.predict(X)

                weights*=np.exp(-stump.alpha*y*preds)  # decreasing the weights of correct classified points while increasing weights of missclassified points        
                weights/=weights.sum() # normalizing weights

                self.stumps.append(stump)
                self.alphas.append(alpha)

                if error<=eps:
                    break

    def predict(self, X):
        scores=np.zeros(X.shape[0])
        for stump in self.stumps:
            scores+=stump.alpha*stump.predict(X)

            y_pred=np.sign(scores)
            y_pred[y_pred==0]=1
        return y_pred



df=pd.read_csv(r"D:\ML\archive (2)\Raisin_Dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

numeric_cols=df.select_dtypes(include=["number"]).columns
categorical_cols=df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()

sns.pairplot(df, vars=numeric_cols, hue="Class", diag_kind="hist")
plt.show()

corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.show()

df.hist(bins=30, figsize=(12, 8))
plt.show()

Q1=df[numeric_cols].quantile(0.25)
Q3=df[numeric_cols].quantile(0.75)
IQR=Q3 - Q1

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

df=df.drop_duplicates()

for col in numeric_cols:
    df[col]=np.clip(df[col], lower[col], upper[col])

encoder=LabelEncoder()
df["Class"]=encoder.fit_transform(df["Class"])  # {0,1}

X=df.drop("Class", axis=1).values
y=df["Class"].values

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train_bin=np.where(y_train==1,1,-1)
y_test_bin=np.where(y_test==1,1,-1)

ada_custom=AdaBoosting(n_estimators=50)
ada_custom.fit(X_train, y_train_bin)
y_pred_custom =ada_custom.predict(X_test)

acc_custom=accuracy_score(y_test_bin, y_pred_custom)
print("Custom AdaBoost Accuracy:", acc_custom)

cm_custom=confusion_matrix(y_test_bin, y_pred_custom)
sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Blues")
plt.title("Custom AdaBoost Confusion Matrix")
plt.show()

ada_sklearn=AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    algorithm="SAMME",
    random_state=42
)

ada_sklearn.fit(X_train, y_train)
y_pred_sklearn = ada_sklearn.predict(X_test)

acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Sklearn AdaBoost Accuracy:", acc_sklearn)

cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
sns.heatmap(cm_sklearn, annot=True, fmt="d", cmap="Greens")
plt.title("Sklearn AdaBoost Confusion Matrix")
plt.show()

print("===================================")
print("Custom AdaBoost  :", acc_custom)
print("Sklearn AdaBoost :", acc_sklearn)
print("===================================")