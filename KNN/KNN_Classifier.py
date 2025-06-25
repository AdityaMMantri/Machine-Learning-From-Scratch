import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
class KNN:
    def __init__(self,k,distance_metric,p=None):
        self.K=k
        self.distance_metric=distance_metric.lower()

        if self.distance_metric=="minkowski":
            if p is None:
                raise ValueError("The value of Hyperparamter p must be given")
            self.p=p
        else:
            self.p=None

    def compute_distance(self,test_data_point,data_point):
        try:
            if self.distance_metric=="euclidean":
                distance=np.sqrt(np.sum((test_data_point - data_point) ** 2))
            elif self.distance_metric=="manhattan":
                distance=np.sum(np.abs(test_data_point-data_point))
            elif self.distance_metric=="minkowski":
                distance = np.sum(np.abs(test_data_point-data_point)**self.p) ** (1/self.p)
            else:
                raise ValueError("Unsupported distance metric try again")
            return distance
        except Exception as e:
            print(f"Error computing distance: {e}")
            return float('inf')
    
    def fit(self,X_train,y_train):
        self.X_train=np.array(X_train)
        self.y_train=np.array(y_train)

    def predict(self,test_data_point):

        distances=[]
        for i,train_point in enumerate(self.X_train):
            dist=self.compute_distance(test_data_point,train_point)
            distances.append((dist,self.y_train[i]))
        
        distances.sort(key=lambda x:x[0]) # sorting the distance in asc order, distances format-> [(value,index)]
        k_nearest=distances[:self.K] # picks the first K tuples

        if self.K%2==1:
            labels=[label for _,label in k_nearest ] # used to extract the labels for the first K tuples
            return Counter(labels).most_common(1)[0][0] # for voting
        else:
            class_weights={}
            for dist,label in k_nearest:
                weight=1/(dist + 1e-5)
                class_weights[label]=class_weights.get(label,0) + weight
            return max(class_weights,key=class_weights.get)

    
    def predictions(self,X_test):
        X_test = np.array(X_test)
        return np.array([self.predict(x) for x in X_test])

iris=load_iris()
X,y=iris.data,iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your custom KNN
model = KNN(k=5, distance_metric="euclidean")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predictions(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

