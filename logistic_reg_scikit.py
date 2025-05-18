import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
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
#--------------------------------------------Logistic regression--------------------------
X=df_scaled.drop(columns=["Purchased"])
y=df_scaled["Purchased"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
