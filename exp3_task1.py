import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\SEM4\ML\Lab_Exp\ML-P3-drinks.csv")
print(df)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="continent", y="total_litres_of_pure_alcohol", palette="Set2")
plt.title("Total Pure Alcohol Consumption by Continent", fontsize=14)
plt.xlabel("Continent", fontsize=12)
plt.ylabel("Litres of Pure Alcohol", fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x="beer_servings",
    y="wine_servings",
    hue="continent",
    size="total_litres_of_pure_alcohol",
    sizes=(20, 200),
    alpha=0.7,
    edgecolor="black",
)
plt.title("Wine vs Beer Servings by Country", fontsize=16)
plt.xlabel("Beer Servings", fontsize=12)
plt.ylabel("Wine Servings", fontsize=12)
plt.legend(title="Continent", loc="upper right", bbox_to_anchor=(1.25, 1))
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()

sns.countplot(data=df,x="continent")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="continent", y="beer_servings", palette="Set2")
plt.title("Beer Servings Distribution by Continent", fontsize=14)
plt.xlabel("Continent", fontsize=12)
plt.ylabel("Beer Servings", fontsize=12)
plt.show()

numeric_cols=df.select_dtypes(include=["number"]).columns
plt.figure(figsize=(8, 6))
corr_matrix=df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="continent", y="total_litres_of_pure_alcohol", palette="Set2")
plt.title("Average Total Pure Alcohol Consumption by Continent", fontsize=14)
plt.xlabel("Continent", fontsize=12)
plt.ylabel("Average Litres of Pure Alcohol", fontsize=12)
plt.show()

