import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_selection import mutual_info_regression,SelectKBest,f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\ML\taxi_trip_pricing.csv")
print(df)
#------------------------------------------------EDA--------------------------------------------------
print(df.info())
print(df.describe())

numeric_cols=df.select_dtypes(include=["number"]).columns
categorical_cols=df.select_dtypes(include=["object"]).columns

for column in numeric_cols:
    plt.figure(figsize=(10,8))
    sns.boxplot(data=df,x=column)
    plt.title(f'Boxplot of {column}', fontsize=14)
    plt.xlabel(column)
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

corr_with_target = df[numeric_cols].corr()["Trip_Price"]
print(corr_with_target)

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

#------------------------------------------------Data Preprocessing-----------------------------------
df=df.drop_duplicates()

numeric_imputer=SimpleImputer(strategy="mean")
df[numeric_cols]=numeric_imputer.fit_transform(df[numeric_cols])

categeory_imputer=SimpleImputer(strategy="constant",fill_value="Unknown")
df[categorical_cols]=categeory_imputer.fit_transform(df[categorical_cols])

encoded_cols=["Traffic_Conditions","Time_of_Day"]
label_encoder=LabelEncoder()
for col in encoded_cols:
    df[col]=label_encoder.fit_transform(df[col])

print(df[encoded_cols])

sns.boxplot(data=df, x="Day_of_Week", y="Trip_Price")
plt.show()
grp=df.groupby("Day_of_Week")["Trip_Price"].mean()
print(grp)

one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_cols_cat = one_hot_encoder.fit_transform(df[["Weather", "Day_of_Week"]])
encoded_col_names = one_hot_encoder.get_feature_names_out(["Weather", "Day_of_Week"])
encoded_df = pd.DataFrame(encoded_cols_cat, columns=encoded_col_names)
df = pd.concat([df, encoded_df], axis=1)
df = df.drop(["Weather", "Day_of_Week"], axis=1)
print(df)
#----------------------------------------------Feature Selection---------------------------------------
X=df.drop(["Trip_Price"],axis=1)
y=df["Trip_Price"]
mutual_score=mutual_info_regression(X,y)
mutual_series = pd.Series(mutual_score, index=X.columns)
mutual_series = mutual_series.sort_values(ascending=False)
print(mutual_series)

f_value,p_value=f_classif(X,y)
top_features = SelectKBest(score_func=f_classif, k=10).fit(X, y)
selected_features = X.columns[top_features.get_support()]
print("Selected Features:", selected_features)
#-----------------------------------------------MLR MODEL-----------------------------------------------
X=df[["Trip_Distance_km","Per_Km_Rate","Trip_Duration_Minutes","Per_Minute_Rate","Traffic_Conditions"]]
y=df["Trip_Price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R² Score: {r2}')

#Mean Squared Error (MSE): 193.90595546494552
#Root Mean Squared Error (RMSE): 13.925011865881677
#R² Score: 0.766480788977096