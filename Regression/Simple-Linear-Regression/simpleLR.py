import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import seaborn as sns
class SimpleLinearRegression:
    def __init__(self,iteration,alpha):
        self.m=0
        self.c=0
        self.alpha=alpha
        self.iteration=iteration
        self.cost_history=[]

    def gradient_descent(self,X,y):
        n=len(X)
        threshold=1e-6
        for i in range(self.iteration):
            y_hat = self.m*X+self.c
            residual=y-y_hat

            cost=(1/n)*np.sum(residual**2)
            self.cost_history.append(cost)

            gradient_slope = (-2/n)*np.sum(residual*X)
            gradient_intercept = (-2/n)*(np.sum(residual))

            step_size_slope=self.alpha*gradient_slope
            step_size_intercept=self.alpha*gradient_intercept
            if abs(step_size_slope) < threshold and abs(step_size_intercept) < threshold:
                print(f"Converged after {i} iterations")
                break

            self.m=self.m-step_size_slope
            self.c=self.c-step_size_intercept

        return self.m,self.c
    def predict_values(self,X):
        y_predict=self.m*X+self.c
        return y_predict
    
    def Plot_regressionline(self,X,y):
        y_predict=self.predict_values(X)
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X, y_predict, color='red', label='Regression line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.show()

    def plot_cost_function(self):
        plt.plot(range(len(self.cost_history)), self.cost_history, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('Cost Function (MSE)')
        plt.title('Cost Function Progress During Gradient Descent')
        plt.show()

    def plot_residuals(self, X, y):
        residuals = y - self.predict_values(X)
        plt.scatter(X, residuals, color='purple', alpha=0.7)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    
    def model_evaluate(self,X,y):
        y_predict=self.predict_values(X)
        r2=r2_score(y,y_predict)
        mse=mean_squared_error(y,y_predict)
        mae=mean_absolute_error(y,y_predict)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared: {r2}")

# m=slope
# c=intercept
# alpha=learning rate
# gradinet vo hota hai jab apan same function ka derivative do alag alag paramter se le jase idhar ek bar m se aur ek bar c se

df=pd.read_csv("YOUR FILEPATH") 
print(df.info())
print(df.describe())

df=df.drop_duplicates()

numeric_cols = df.select_dtypes(include=["number"]).columns

for column in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=column, palette="Set3")
    plt.title(f'Boxplot of {column}', fontsize=14)
    plt.xlabel(column)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True)
    plt.show()

null_count_num = df[numeric_cols].isnull().sum()
plt.figure(figsize=(12, 6))
plt.barh(null_count_num.index, null_count_num.values, color="skyblue")
plt.title('Number of Null Values per Numeric Column', fontsize=16)
plt.ylabel('Numeric Columns', fontsize=12)
plt.xlabel('Number of Null Values', fontsize=12)
plt.grid(True)
plt.show()

df[numeric_cols].hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms for Numeric Columns', fontsize=16)
plt.show()

corr_matrix=df[numeric_cols].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

corr_with_target = df[numeric_cols].corr()["TARGET(PRICE_IN_LACS)"]
print(corr_with_target)

Q1=df[numeric_cols].quantile(0.25)
Q3=df[numeric_cols].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1- 1.5 * IQR
upper_bound=Q3+ 1.5 * IQR
outliers = (df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)
print("Number of outliers per column:")
print(outliers.sum())

print(df["BHK_OR_RK"].value_counts())

#-------------------------------pre processing---------------------------
for col in numeric_cols:
    df[col] = np.clip(df[col], lower_bound[col], upper_bound[col])
outliers = (df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)   
print(outliers.sum())


scaler=RobustScaler()
df[numeric_cols]=scaler.fit_transform(df[numeric_cols])

print(df[numeric_cols].head(50))
print(df[numeric_cols].describe())  
     
X=df["SQUARE_FT"]
y=df["TARGET(PRICE_IN_LACS)"]


#model = SimpleLinearRegression(iteration=1000, alpha=0.001)
#model.gradient_descent(X, y)
#model.Plot_regressionline(X, y)
#model.plot_cost_function()
#model.plot_residuals(X,y)

    # Make predictions
#predictions = model.predict_values(X)
#print("Predicted values:", predictions)
#model.model_evaluate(X,y)
#Mean Squared Error: 0.5394948438263107
#Mean Absolute Error: 0.5808984695100653
#R-squared: 0.304367654353334

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SimpleLinearRegression(iteration=10000, alpha=0.001)
model.gradient_descent(X_train, y_train)
model.Plot_regressionline(X_train, y_train)
model.plot_cost_function()
model.plot_residuals(X_train, y_train)

model.Plot_regressionline(X_test, y_test)
model.model_evaluate(X_test, y_test)

predictions = model.predict_values(X_test)
print("Predicted values for test set:", predictions)


