import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression,SelectKBest,f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\ML\manufacturing.csv")
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
#--------------------------------------------------Polynomial Model----------------------------------------
X=df[["Temperature (°C)"]]
y=df["Quality Rating"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

poly_model=PolynomialFeatures(degree=3)
X_train_poly=poly_model.fit_transform(X_train)
X_poly_test = poly_model.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_poly_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

X_train_sorted_indices = X_train[:, 0].argsort()  # Sorting by the first column if X_train is 2D
X_train_sorted = X_train[X_train_sorted_indices]
y_train_sorted_pred = model.predict(poly_model.transform(X_train_sorted))  # Predict for sorted X_train

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data Points')  # Scatter plot of training data
plt.plot(X_train_sorted, y_train_sorted_pred, color='red', label='Regression Line')  # Regression line
plt.xlabel('X (Temperature °C)')
plt.ylabel('y (Quality Rating)')
plt.legend()
plt.show()

