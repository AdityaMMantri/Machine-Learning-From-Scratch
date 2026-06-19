from logisticR import BinaryLogisticRegression
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score)

class OneVsRestLogistic:
    def __init__(self,learning_rate,iteration,regularization=None,_lambda_=0.01,l1_ratio=0.5,init="random"):
        self.learning_rate=learning_rate
        self.iteration=iteration
        # Dictionary to store trained binary classifiers to later easliy access them for prediction
        # {   0: model_for_class_0,
        #     1: model_for_class_1,
        #     2: model_for_class_2   } type hinting is done for future use
        self.classifiers:dict[int, BinaryLogisticRegression]={}
        # Stores all unique class labels found during training
        self.classes=None
        self.init=init

        # regularization parameters
        self.regularization=regularization
        self._lambda_=_lambda_
        self.l1_ratio=l1_ratio
    
    def fit(self,X,y):
        # input-> X_train and y_train
        # output-> self.classifiers filled and self.classes
        # algorithm of fit function->
        # 1. Find all unique classes.
        # 2. For each class:
            # a) Treat that class as Positive (1)
            # b) Treat every other class as Negative (0)
            # c) Train a Binary Logistic Regression model
            # d) Store the trained model
        # 3. Repeat until every class has its own binary classifier.

        self.classes=np.unique(y)
        for current_class in self.classes:
            print(f"Training classifier for class {current_class}")
            y_binary_labels=(y==current_class).astype(int)
            model=BinaryLogisticRegression(learning_rate=self.learning_rate,iteration=self.iteration)
            model.gradient_descent(X=X,y=y_binary_labels)
            self.classifiers[current_class]=model

    def predict_probabilities(self,X):
        # input -> X_test or any new unseen data, output -> probability matrix of shape: (n_samples, n_classes)
        # Algorithm:
        # 1. Find the number of samples and classes.
        # 2. Create an empty probability matrix:
        #       Rows    -> Samples
        #       Columns -> Classes
        # 3. Loop through each trained classifier.
        # 4. Get probability predictions from the current classifier.
        # 5. Store those probabilities in the corresponding class column.
        # 6. Repeat until probabilities for all classes are stored.

        n_samples=len(X)
        n_classes=len(self.classes)
        all_probabilities=np.zeros((n_samples,n_classes))

        for idx,current_class in enumerate(self.classes):
            model=self.classifiers[current_class]
            probabilities=model.predict_values(X=X) # basically it will go sample wise for each sample it will use model 0 then model 1...
            all_probabilities[:,idx] = probabilities
        return all_probabilities

    def predict(self,X):
        # for each sample find the class with the highest probablity 
        # find the class index and return the actual class label
        probability_matrix=self.predict_probabilities(X)
        class_indices=np.argmax(probability_matrix,axis=1)
        predictions=self.classes[class_indices] # works with any class labels not just 0,1,2

        return predictions
            

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

my_model = OneVsRestLogistic(learning_rate=0.01,iteration=5000,regularization="l1",init="random")
my_model.fit(X_train,y_train)
my_pred = my_model.predict(X_test)

print("="*50)
print("MY OVR LOGISTIC REGRESSION")
print("="*50)

print("Accuracy:")
print(accuracy_score(y_test,my_pred))

print("\nF1 Score (Macro):")
print(f1_score(y_test,my_pred,average='macro'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,my_pred))

print("\nClassification Report:")
print(classification_report(y_test,my_pred))


sk_model = OneVsRestClassifier(LogisticRegression(max_iter=5000))
sk_model.fit(X_train,y_train)
sk_pred = sk_model.predict(X_test)

print("="*50)
print("SCIKIT-LEARN OVR")
print("="*50)

print("Accuracy:")
print(accuracy_score(y_test,sk_pred))

print("\nF1 Score (Macro):")
print(f1_score(y_test,sk_pred,average='macro'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,sk_pred))

print("\nClassification Report:")
print(classification_report(y_test,sk_pred))

print("\nMY MODEL WEIGHTS")

for cls in my_model.classes:

    model = my_model.classifiers[cls]
    print(f"\nClass {cls}")
    print("Weights:")
    print(model.weights)
    print("Bias:")
    print(model.bias)

print("\nSKLEARN WEIGHTS")

for idx, estimator in enumerate(sk_model.estimators_):
    print(f"\nClass {idx}")
    print("Weights:")
    print(estimator.coef_)
    print("Bias:")
    print(estimator.intercept_)