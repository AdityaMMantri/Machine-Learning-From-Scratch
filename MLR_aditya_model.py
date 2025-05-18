import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.feature_selection import mutual_info_regression,SelectKBest,f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
class MultipleLinearRegression:
    def __init__(self,iterations,alpha,feature_names):
        self.alpha=alpha
        self.iterations=iterations
        self.weights=0 # works as slope just it is an array of features/attributes
        self.bias=0 # it is the B0,intercept
        self.feature_names=feature_names
    
    def gradient_descent(self,X,y):
        n_sample,n_feature=X.shape
        self.weights=np.zeros(n_feature) #jitne number of columns utne feautres to utna array size chaiye humko
        threshold=1e-8
        for i in range(self.iterations):
            y_hat=np.dot(X,self.weights) + self.bias
            residual=y-y_hat

            gradient_weights=(-2/n_sample)*np.dot(X.T,residual)
            gradient_bias=(-2/n_sample)*np.sum(residual)

            step_size_weights=self.alpha*gradient_weights
            step_size_bias=self.alpha*gradient_bias

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

    def visualize(self,X,y):    
        X_df=pd.DataFrame(X,columns=self.feature_names)
        X_df["Target"]=y
        sns.pairplot(X_df,diag_kind="hist", hue="Target")
        plt.show()

        n_len=len(self.feature_names)
        n_col=2
        n_rows=(n_len+1)//2

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, figsize=(12, n_rows * 6))
        fig.suptitle("Residuals vs. Features", fontsize=16)
        axes = axes.flatten()
        y_predict=self.predict_values(X)
        residual=y-y_predict

        for i,feature in enumerate(self.feature_names):
            axes[i].scatter(X[feature],residual,color='blue', alpha=0.6)
            axes[i].axhline(y=0, color='red', linestyle='--', lw=2)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Residuals")
            axes[i].grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(residual, kde=True, color='blue', bins=30, stat='density')
        plt.title("Histogram of Residuals", fontsize=16)
        plt.xlabel("Residuals", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True)
        plt.show()

# Equation-> B0 + B1*x1 + B2*x2.......
# again our aim is to find such values of weights and bias such that the sum of the residuals is minimums so that cost function can be minimized
# the cost funciton is the same as SLR just the difference here is we have an array of independent varables
#y_hat=X⋅w+b
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
#-------------------------------------------------MLR model-------------------------------------------
X=df[["Trip_Distance_km","Per_Km_Rate","Trip_Duration_Minutes","Per_Minute_Rate","Traffic_Conditions"]]
y=df["Trip_Price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
mlr_model = MultipleLinearRegression(iterations=1500000, alpha=0.0001, feature_names=X.columns)
mlr_model.gradient_descent(X_train.values, y_train.values)
mlr_model.predict_values(X_test)

# Evaluate the model
mlr_model.evalute_model(X_test.values, y_test.values)