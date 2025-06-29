import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression,SelectKBest,f_classif
from sklearn.model_selection import train_test_split
class SimplePolynomialRegression:
    def __init__(self,alpha,iterations,degree):
        self.alpha=alpha
        self.iterations=iterations
        self.degree=degree
        self.coefficients=np.zeros(degree+1)
        self.cost_history = []
    
    def design_matrix(self,X):
        design_matrix_model=np.vstack([X**i for i in range(0,self.degree+1)])
        design_matrix=design_matrix_model.T
        return design_matrix
    
    def gradient_descent(self, y, X):
        design_matrix = self.design_matrix(X)
        n = len(X)
        threshold = 1e-6 

        for i in range(self.iterations):
            y_hat=np.dot(design_matrix, self.coefficients) 
            residual=y-y_hat  

            cost=(1/n) * np.sum(residual**2)
            self.cost_history.append(cost)

            gradient=(-2/n) * np.dot(design_matrix.T, residual)
            step_size=self.alpha * gradient

            if np.linalg.norm(step_size) < threshold: 
                print(f"Converged after {i} iterations")
                break

            self.coefficients=self.coefficients-step_size 

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost}")
    
    def predict_value(self,X):
        design_matrix=self.design_matrix(X)
        y_predicted=np.dot(design_matrix,self.coefficients)
        return y_predicted

    def plot_regression_line(self, X, y):
        y_predict = self.predict_value(X)
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(sorted(X), self.predict_value(np.array(sorted(X))), color='red', label='Regression line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Regression (Degree: {self.degree})')
        plt.legend()
        plt.show()

    def plot_cost_function(self):
        plt.plot(range(len(self.cost_history)), self.cost_history, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function (MSE)')
        plt.title('Cost Function Progress During Gradient Descent')
        plt.show() 

    def model_evaluate(self, X, y):
        y_predict=self.predict_value(X)
        r2=r2_score(y, y_predict)
        mse=mean_squared_error(y, y_predict)
        mae=mean_absolute_error(y, y_predict)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")

df=pd.read_csv(File Path)
#-------------------------------------------EDA----------------------------------------------------
print(df)
print(df.info())
print(df.describe())

for column in df:
    plt.figure(figsize=(12,8))
    sns.boxplot(data=df,x=column)
    plt.title(f'Boxplot of {column}', fontsize=14)
    plt.xlabel(column)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True)
    plt.show()

sns.pairplot(data=df,diag_kind="hist")
plt.show()

corr_matrix=df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

corr_with_target = df.corr()["Quality Rating"]
print(corr_with_target)

null_count=df.isna().sum()
print(null_count)
null_count.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Count of Missing Values per Column', fontsize=14)
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Count of Missing Values', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1- 1.5 * IQR
upper_bound=Q3+ 1.5 * IQR
outliers = (df<lower_bound) | (df>upper_bound)
print("Number of outliers per column:")
print(outliers.sum())

df.hist(bins=30,figsize=(12,8))
plt.show()

print(f"the kurtosis of the data is:{df.kurtosis()}")
print(f"the skewness of the data is:{df.skew()}")
#----------------------------------------------------Feature Selection-------------------------------------
X=df.drop(["Quality Rating"],axis=1)
y=df["Quality Rating"]
mutual_score=mutual_info_regression(X,y)
mutual_series = pd.Series(mutual_score, index=X.columns)
mutual_series = mutual_series.sort_values(ascending=False)
print(mutual_series)
#-------------------------------------------------------Data Preprocessing---------------------------------

df=df.drop_duplicates()

scaler=MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print(df_scaled)
#----------------------------------------------------Feature Selection-------------------------------------
f_value,p_value=f_classif(X,y)
top_features = SelectKBest(score_func=f_classif, k=10).fit(X, y)
selected_features = X.columns[top_features.get_support()]
print("Selected Features:", selected_features)

X = df_scaled["Temperature (°C)"].values
y = df_scaled["Quality Rating"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1)
X_test = X_test.reshape(-1)

model = SimplePolynomialRegression(alpha=0.01, iterations=1000000, degree=3)
model.gradient_descent(y_train, X_train)

print("Training Evaluation:")
model.model_evaluate(X_train, y_train)

print("Testing Evaluation:")
model.model_evaluate(X_test, y_test)

model.plot_regression_line(X_test, y_test)

model.plot_cost_function()

