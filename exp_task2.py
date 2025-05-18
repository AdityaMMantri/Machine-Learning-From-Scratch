import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\SEM4\ML\Lab_Exp\ML-P3-seeds.csv")
print(df)

print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.dtypes)
print(df.shape)
print(df.columns)
print(df.isna().sum())
for cols in df.columns:
    print(f" the unique value of {cols} is: {df[cols].unique()}")
    
plt.figure(figsize=(10,5))
plt.scatter(df["Kernel.Width"],df["Kernel.Length"])
plt.title("Scatter Plot of Kernel Width vs Kernel Length", fontsize=14)
plt.xlabel("Kernel Width", fontsize=12)
plt.ylabel("Kernel Length", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

sns.set(style="whitegrid")
joint_plot =sns.jointplot(
    data=df,
    x="Perimeter",
    y="Compactness",
    kind="scatter", 
    height=8,
    space=0.5,
    color="purple"
)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df,x="Compactness",palette="Set3")
plt.title(f'Boxplot of {"Compactness"}',fontsize=14)
plt.xlabel("Compactness")
plt.ylabel('Value', fontsize=12)
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Perimeter", palette="Set3")
plt.title(f'Boxplot of {"Perimeter"}', fontsize=14)
plt.xlabel("Perimeter")
plt.ylabel('Value', fontsize=12)
plt.grid(True)
plt.show()

numeric_cols=df.select_dtypes(include=["number"]).columns
sns.pairplot(data=df[numeric_cols],diag_kind="hist")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="Type",y="Compactness", palette="muted")
plt.title("Violin Plot of Compactness by Type", fontsize=14)
plt.xlabel("Type",fontsize=12)
plt.ylabel("Compactness",fontsize=12)
plt.grid(axis="y",linestyle="--", alpha=0.6)
plt.show()

sns.countplot(data=df,x="Type",palette="Set1")
plt.show()

sns.pairplot(data=df[numeric_cols],diag_kind="kde",palette="Set1",height=2.5)
plt.show()