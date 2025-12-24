from CART_RANDOM import Decision_tree
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

class Random_Forest:
    def __init__(self,n_trees,max_depth,min_sample_split,feature=None):
        self.number_of_tree=n_trees
        self.max_depth=max_depth
        self.min_sample_split=min_sample_split
        self.feature=feature
        self.trees=[]

# number_of_tree-> how many decision trees are to be built
# max_depth-> for each tree the maximum depth level
# min_sample_split-> Minimum number of samples required to split a node in a decision tree. If a node has fewer samples, it becomes a leaf.
# feature-> Number of random features to consider at each split.
# trees-> to store each individual tree.
    def fit(self,X,y):
        self.trees=[]
        for i in range(self.number_of_tree):
            X_data,y_data=self.bootstrapping(X,y)
            tree=Decision_tree(max_depth=self.max_depth,min_sample_split=self.min_sample_split)
            tree.fit(X_data,y_data)
            self.trees.append(tree)
    
    def bootstrapping(self,X,y):
        n_samples=len(X)
        indices=np.random.choice(n_samples,n_samples,replace=True)
        return X[indices],y[indices]

    def predict(self,X):
        single_tree_pred=np.array([tree.predict(X) for tree in self.trees])
            # Step 2: Transpose it to shape (n_samples, n_trees)
        tree_predictions = np.swapaxes(single_tree_pred, 0, 1)

        # Step 3: For each sample, do majority voting
        final_predictions = [self._most_common_label(preds) for preds in tree_predictions]
        return np.array(final_predictions)

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]


df=pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\ML\drug200.csv")
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

encoded_cols=["Sex","BP","Cholesterol","Drug"]
label_encoder=LabelEncoder()    
for col in encoded_cols:
    df[col]=label_encoder.fit_transform(df[col])

print(df[encoded_cols])

print(df)

X=df.drop("Drug",axis=1)
y=df["Drug"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Step 1: Create an instance of your custom Random Forest
rf = Random_Forest(n_trees=100, max_depth=5, min_sample_split=2)

# Step 2: Train (fit) the model
rf.fit(X_train.values, y_train.values)

# Step 3: Predict on the test set
y_pred = rf.predict(X_test.values)

# Step 4: Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
