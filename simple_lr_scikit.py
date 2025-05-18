import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\train.csv")
#---------------------------------EDA--------------------------------------
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

#--------------------------------Linear regression-------------------------

X=df[["SQUARE_FT"]]
y=df["TARGET(PRICE_IN_LACS)"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

model=LinearRegression()

model.fit(X_train,y_train)

y_predict=model.predict(X_test)
print(y_predict)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data Points')  # Scatter plot for actual data points
plt.plot(X_test, y_predict, color='red', label='Regression Line')  # Regression line
plt.title('Linear Regression: SQUARE_FT vs TARGET(PRICE_IN_LACS)', fontsize=16)
plt.xlabel('SQUARE_FT', fontsize=14)
plt.ylabel('TARGET(PRICE_IN_LACS)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

#Mean Squared Error: 0.5236893062899539
#R² Score: 0.31308936504830365