import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

class NaiveBayerClassifier:
    def __init__(self,var_smoothing=1e-9):
        self.mean=None
        self.variance=None
        self.prior=None
        self.classes=None
        self.var_smoothing=var_smoothing

    def _compute_statistics(self,X,y):
        X=np.asarray(X, dtype=np.float64)
        y=np.array(y)
        self.classes=np.unique(y)
        n_classes=len(self.classes)
        n_features=X.shape[1]

        self.mean=np.zeros((n_classes,n_features),dtype=np.float64)
        self.variance=np.zeros((n_classes,n_features),dtype=np.float64)
        self.prior=np.zeros(n_classes,dtype=np.float64)

        for index,Class in enumerate(self.classes):
            X_class=X[y==Class]
            self.mean[index]=X_class.mean(axis=0)
            self.variance[index]=X_class.var(axis=0)
            self.prior[index]=X_class.shape[0] / X.shape[0]

        epsilon=self.var_smoothing * np.var(X, axis=0).max()
        self.variance+=epsilon
        
        print("Mean:\n", self.mean)
        print("Variance:\n", self.variance)
        print("Prior:\n", self.prior)

    def fit(self, X, y):
        X=np.asarray(X, dtype=np.float64)
        y=np.asarray(y)

        if X.ndim!=2:
            raise ValueError("X must be a 2-dimensional array.")

        if y.ndim!=1:
            raise ValueError("y must be a 1-dimensional array.")

        if len(X)!=len(y):  
            raise ValueError("X and y must contain the same number of samples.")
        
        if len(np.unique(y))<2:
            raise ValueError("At least two classes are required.")

        self._compute_statistics(X, y)
        return self
    
    def _log_gaussian_pdf(self,X,class_index):
        mean = self.mean[class_index]
        variance = self.variance[class_index]

        log_coefficient=-0.5 * np.log(2*np.pi * variance)
        log_exponent=-((X - mean)**2) / (2*variance)
        return log_coefficient + log_exponent
    
    def predict(self, X):
        if self.mean is None:
            raise ValueError("This NaiveBayesClassifier instance is not fitted yet.Call 'fit()' before using predict().")
        X = np.asarray(X, dtype=np.float64)
        y_pred=[]
        if X.shape[1] != self.mean.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, but the model was trained with {self.mean.shape[1]} features.")

        for x in X:
            posteriors=[]
            for class_index,class_label in enumerate(self.classes):
                prior=np.log(self.prior[class_index])
                conditional_prob=np.sum(self._log_gaussian_pdf(x,class_index))
                posterior=prior + conditional_prob
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)
    
if __name__=="__main__":
    df=pd.read_csv(r"F:\Machine-Learning-From-Scratch\Naive_Bayes\Gaussian_NB\Naive-Bayes-Classification-Data.csv")
    print(df)
    print(df.head())
    print(df.describe())
    print(df.info())

    numeric_cols=df.select_dtypes(include=["number"])
    corr_matrix=numeric_cols.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title('Correlation Matrix')

    null_count_num=numeric_cols.isnull().sum()
    plt.figure(figsize=(12,8))
    plt.barh(null_count_num.index,null_count_num.values,color="skyblue")
    plt.ylabel('Numeric Columns')
    plt.xlabel('Number of Null Values')
    plt.title('Number of Null Values per Numeric Column')
    plt.xticks(rotation=45)
    plt.grid(True)

    for coloum in df:
        plt.figure()
        df.boxplot([coloum])
        plt.title(f'Boxplot of {coloum}', fontsize=14)  
        plt.ylabel('Value', fontsize=12)
        plt.grid(True) 

    sns.pairplot(data=df,diag_kind="hist")
    plt.show()

    df.hist(bins=10, figsize=(10, 5))
    plt.show()

    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1- 1.5 * IQR
    upper_bound=Q3+ 1.5 * IQR
    outliers = (df < lower_bound) | (df > upper_bound)
    print("Number of outliers per column:")
    print(outliers.sum())

    df=df.drop_duplicates()

    X=df[["glucose","bloodpressure"]]
    y=df["diabetes"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    nb=NaiveBayerClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Custom Model Accuracy:", accuracy)

    print("="*50)
    print("MY PARAMETERS")
    print("="*50)

    print("Classes:")
    print(nb.classes)

    print("\nPrior:")
    print(nb.prior)

    print("\nMean:")
    print(nb.mean)

    print("\nVariance:")
    print(nb.variance)

    model=GaussianNB()
    model.fit(X_train,y_train)
    y_pred_sk=model.predict(X_test)
    accuracy_sk = accuracy_score(y_test, y_pred_sk)
    print("Scikit-learn's Accuracy:", accuracy_sk)

    print("="*50)
    print("SCIKIT-LEARN PARAMETERS")
    print("="*50)

    print("Classes:")
    print(model.classes_)

    print("\nPrior:")
    print(model.class_prior_)

    print("\nMean:")
    print(model.theta_)

    print("\nVariance:")
    print(model.var_)