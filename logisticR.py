import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
class LogisticRegression:
    def __init__(self,alpha,iteration,feature_names):
        self.alpha=alpha
        self.iterations=iteration
        self.weights=0
        self.bias=0
        self.feature_names=feature_names
    
    def sigmoid_probability(self, z):
        return 1/ (1 + np.exp(-z)) 
    
    def compute_cost(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
        m=y.shape[0]
        cost=-(1 / m) * np.sum(y*np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return cost

    
    def gradient_descent(self,X,y):
        n_sample,n_feature=X.shape
        self.weights=np.zeros(n_feature) #jitne number of columns utne feautres to utna array size chaiye humko
        threshold=1e-4
        costs=[]

        for i in range(self.iterations):
            z=np.dot(X,self.weights) + self.bias
            y_pred=self.sigmoid_probability(z)
            cost = self.compute_cost(y, y_pred)
            costs.append(cost)
            gradient_weight = (1 / n_sample) * np.dot(X.T, (y_pred - y))
            gradient_bias = (1 / n_sample) * np.sum(y_pred - y)


            step_size_weight=self.alpha*gradient_weight
            step_size_bias=self.alpha*gradient_bias

            if np.all(step_size_weight) < threshold and np.abs(step_size_bias) < threshold:
                print(f"Converged after {i} iterations")
                break

            self.weights=self.weights-step_size_weight
            self.bias=self.bias-step_size_bias

            if i % 100==0 or i==self.iterations-1:
                cost=self.compute_cost(y,y_pred)
                print(f"Iteration {i}, Cost: {cost}")

    def predict_values(self,X):
        z=np.dot(X, self.weights) + self.bias
        return self.sigmoid_probability(z) 
    
    def predict_binary(self, X):
        probabilities=self.predict_values(X)  
        return (probabilities >= 0.5).astype(int) 
    
    def model_accuracy(self, X, y):
        y_pred=self.predict_binary(X) 
        correct_predictions=np.sum(y_pred == y) 
        return (correct_predictions/len(y)) * 100

    def plot_comparison(self, X, y):
        y_pred = self.predict_binary(X)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(y)), y, label='Actual Values', marker='o', linestyle='--', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label='Predicted Values', marker='x', linestyle='-', color='red')
        plt.xlabel('Sample Index')
        plt.ylabel('Class Label')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sigmoid_curve_probability(self,X,y):
        z=np.dot(X, self.weights) + self.bias
        sigmoid_value=self.sigmoid_probability(z)
        plt.plot(z,sigmoid_value)
        plt.plot(z,sigmoid_value,color='green',label='Sigmoid Curve',linewidth=2)
        plt.scatter(z,sigmoid_value, color='red', label='Predicted Values', zorder=5)
        plt.axvline(0,color='black', linewidth=1)# Line at z = 0
        plt.axhline(0.5, color='black', linewidth=1)# Line at y = 0.5
        plt.xlabel('Z Value (Input to Sigmoid)')
        plt.ylabel('Sigmoid Output')
        plt.title('Sigmoid Curve with Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\ML\log_reg.csv")
print(df)
#--------------------------------EDA-----------------------------------
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

sns.pairplot(data=df[numeric_cols],diag_kind="hist")
plt.show()

corr_matrix=df[numeric_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

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
#---------------------------------------data preprocessing--------------------------------
df=df.drop_duplicates()
 
numeric_imputer=SimpleImputer(strategy="mean")
df[numeric_cols]=numeric_imputer.fit_transform(df[numeric_cols])

categeory_imputer=SimpleImputer(strategy="constant",fill_value="Unknown")
df[categorical_cols]=categeory_imputer.fit_transform(df[categorical_cols])


one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_cols_cat = one_hot_encoder.fit_transform(df[["Gender"]])
encoded_col_names = one_hot_encoder.get_feature_names_out(["Gender"])
encoded_df = pd.DataFrame(encoded_cols_cat, columns=encoded_col_names)
df = pd.concat([df, encoded_df], axis=1)
df = df.drop(["Gender","User ID"], axis=1)
print(df)

df=df.drop_duplicates()

scaler=MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print(df_scaled)
#-------------------------------------------Logistic Regression----------------------
X=df_scaled.drop(columns=["Purchased"])
y=df_scaled["Purchased"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
feature_names = ["Age", "EstimatedSalary", "Gender_Male"]
log_reg = LogisticRegression(alpha=0.01, iteration=1000000, feature_names=feature_names)

log_reg.gradient_descent(X_train.values, y_train.values)

accuracy_train = log_reg.model_accuracy(X_train.values, y_train.values)
accuracy_test = log_reg.model_accuracy(X_test.values, y_test.values)

print(f"Training Accuracy: {accuracy_train:.2f}%")
print(f"Test Accuracy: {accuracy_test:.2f}%")

log_reg.plot_comparison(X_test.values, y_test.values)
y_pred = log_reg.predict_binary(X_test.values)

log_reg.sigmoid_curve_probability(X_test.values, y_test.values)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
y_train_pred = log_reg.predict_binary(X_train.values)
print("Training Predictions:", np.unique(y_train_pred, return_counts=True))
y_test_pred = log_reg.predict_binary(X_test.values)
print("Test Predictions:", np.unique(y_test_pred, return_counts=True))

