import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
class KNNRegressor:
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

        values=[target for _,target in k_nearest]
        return np.mean(values)
    
    def predictions(self,X_test):
        X_test = np.array(X_test)
        return np.array([self.predict(x) for x in X_test])
    
X,y=load_diabetes(return_X_y=True) # returns X and y as numpy array
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model=KNNRegressor(k=5,distance_metric="euclidean")
model.fit(X_train,y_train)
y_pred=model.predictions(X_test)

KNN_scikit=KNeighborsRegressor(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
KNN_scikit.fit(X_train,y_train)
y_pred_sci=KNN_scikit.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

print("RMSE:", mean_squared_error(y_test, y_pred_sci))
print("R² Score:", r2_score(y_test, y_pred_sci))

