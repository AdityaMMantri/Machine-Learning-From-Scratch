import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
class LassoRegression:
    def __init__(self,alpha,iterations,Lambda,feature_name):
        self.alpha=alpha
        self.iterations=iterations
        self.Lambda=Lambda
        self.weights=0
        self.bias=0
        self.feature_names=feature_name
        self.cost_history=[]

    def cost_function(self,X,y):
        n_sample = X.shape[0]
        y_hat = np.dot(X, self.weights) + self.bias
        residual = y - y_hat
        cost = (1/n_sample) * np.sum(residual**2) + self.Lambda * np.sum(np.abs(self.weights))  # L1 norm for Lasso
        return cost
    
    def gradient_descent(self,X,y):
        n_sample, n_feature=X.shape
        self.weights=np.zeros(n_feature)  # Initialize weights to zeros
        threshold=1e-8

        for i in range(self.iterations):
            y_hat=np.dot(X, self.weights) + self.bias
            residual=y-y_hat

            gradient_weights=(-2/n_sample) * np.dot(X.T, residual) + self.Lambda * np.sign(self.weights)  # L1 regularization gradient using sigh function as mod(slope)is not differnatable
            gradient_bias=(-2/n_sample) * np.sum(residual)

            step_size_weights=self.alpha*gradient_weights
            step_size_bias=self.alpha*gradient_bias

            cost=self.cost_function(X, y)
            self.cost_history.append(cost)

            if np.linalg.norm(step_size_weights) < threshold and abs(step_size_bias) < threshold:
                print(f"Converged after {i} iterations")
                break

            self.weights=self.weights-step_size_weights
            self.bias=self.bias-step_size_bias
        
    def predict_values(self,X):
        y_predict=np.dot(X,self.weights)+self.bias
        return y_predict
    
    def evalute_model(self,X,y):
        y_predict=self.predict_values(X)
        r2=r2_score(y,y_predict)
        mse=mean_squared_error(y,y_predict)
        mae=mean_absolute_error(y,y_predict)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")

    def plot_cost_function(self):
        plt.plot(range(len(self.cost_history)), self.cost_history, label='Cost Function')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function over Iterations')
        plt.legend()
        plt.show()

df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\archive (1)\train.csv")
print(df)
print(df.head())
print(df.info())
print(df.describe())

numeric_cols=df.select_dtypes(include=["number"]).columns
categorical_cols=df.select_dtypes(include=["object"]).columns

for coloumn in numeric_cols:
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df,x=coloumn)
    plt.title(f'Boxplot of {coloumn}', fontsize=14)
    plt.xlabel(coloumn)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True)
    plt.show()

#sns.pairplot(data=df[numeric_cols],diag_kind="hist")
#plt.show()

#corr_matrix=df[numeric_cols].corr()
#plt.figure(figsize=(12,8))
#sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",linewidths=0.5)
#plt.title('Correlation Matrix')
#plt.show()

correlation_with_target = df[numeric_cols].corr()['SalePrice'].sort_values(ascending=False)
print(correlation_with_target)

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df.hist(bins=30,figsize=(12,8))
plt.show()

null_count=df[numeric_cols].isna().sum()
print(null_count)
null_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Count of Missing Values per Column', fontsize=14)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Count of Missing Values', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

Q1=df[numeric_cols].quantile(0.25)
Q3=df[numeric_cols].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1- 1.5 * IQR
upper_bound=Q3+ 1.5 * IQR
outliers = (df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)
print("Number of outliers per column:")
print(outliers.sum())

df=df.drop_duplicates()
 
numeric_imputer=SimpleImputer(strategy="mean")
df[numeric_cols]=numeric_imputer.fit_transform(df[numeric_cols])

categeory_imputer=SimpleImputer(strategy="constant",fill_value="Unknown")
df[categorical_cols]=categeory_imputer.fit_transform(df[categorical_cols])

unique_values={col: df[col].nunique() for col in categorical_cols}
label_encoder=LabelEncoder()
df_label_encoded=df.copy()

for col in categorical_cols:
    if unique_values[col] <= 9:
        df_label_encoded[col]=label_encoder.fit_transform(df_label_encoded[col].astype(str))

remaining_categorical_cols=[col for col in categorical_cols if col not in df_label_encoded.columns or unique_values[col] > 9]
df_one_hot_encoded = pd.get_dummies(df[remaining_categorical_cols], drop_first=True)

df_final=pd.concat([df_label_encoded, df_one_hot_encoded, df[numeric_cols]], axis=1)
df_final=df_final.loc[:, ~df_final.columns.duplicated()]  
print(df_final.head(20))
print(f"Final DataFrame shape: {df_final.shape}")

df_final=df_final.drop_duplicates()

X=df_final.drop(columns=["SalePrice"],axis=1)
X=pd.get_dummies(X, drop_first=True)
y=df_final["SalePrice"]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

alpha=0.001
iterations=100000
Lambda=0.1
feature_names=X_train.columns

lasso_model=LassoRegression(alpha=alpha, iterations=iterations, Lambda=Lambda, feature_name=feature_names)
lasso_model.gradient_descent(X_train_scaled, y_train.values)

print("\nModel Evaluation on Training Set:")
lasso_model.evalute_model(X_train_scaled, y_train.values)

print("\nModel Evaluation on Test Set:")
lasso_model.evalute_model(X_test_scaled, y_test.values)

y_pred = lasso_model.predict_values(X_test_scaled)
    

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted SalePrice')
plt.grid(True)
plt.show()
