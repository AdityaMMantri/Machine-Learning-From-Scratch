import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayerClassifier:
    def __init__(self):
        self.mean=0
        self.variance=0
        self.prior=0
        self.classes=None

    def Prior_Probability(self,X,y):
        X=np.array(X)
        y=np.array(y)
        self.classes=np.unique(y)
        n_classes=len(self.classes)
        n_features=X.shape[1]

        self.mean=np.zeros((n_classes,n_features),dtype=np.float64)
        self.variance=np.zeros((n_classes,n_features),dtype=np.float64)
        self.prior=np.zeros(n_classes,dtype=np.float64)

        for index,Class in enumerate(self.classes):
            X_class_df=X[y==Class]
            self.mean[index]=X_class_df.mean(axis=0)
            self.variance[index]=X_class_df.var(axis=0)
            self.prior[index]=X_class_df.shape[0] / X.shape[0]
        
        print("Mean:\n", self.mean)
        print("Variance:\n", self.variance)
        print("Prior:\n", self.prior)

    def fit(self, X, y):
        self.Prior_Probability(X, y)
    
    def Gaussian_conditonal_probability(self,X,y):
        eps=1e-6
        exponent=np.exp(-0.5 * ((X-self.mean)**2) / (self.variance + eps))
        return (1/np.sqrt(2*np.pi*(self.variance + eps))) * exponent
    
    def predict(self, X):
        X=np.array(X)
        y_pred=[]

        for x in X:
            posteriors=[]
            for index,Class in enumerate(self.classes):
                prior=np.log(self.prior[index])
                conditional_prob=np.sum(np.log(self.Gaussian_conditonal_probability(x, index)))
                posterior=prior + conditional_prob
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\ML\Naive-Bayes-Classification-Data.csv")
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
print("Model Accuracy:", accuracy)