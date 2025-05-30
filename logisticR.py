import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
class LogisticRegression:
    def __init__(self,alpha,iteration,feature_names):
        self.alpha=alpha
        self.iterations=iteration
        self.weights=0
        self.bias=0
        self.feature_names=feature_names
    
    def sigmoid_probability(self, z):
        return 1/ (1 + np.exp(-z)) 
    
    def compute_cost(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
        m=y.shape[0]
        cost=-(1 / m) * np.sum(y*np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return cost

    
    def gradient_descent(self,X,y):
        n_sample,n_feature=X.shape
        self.weights=np.zeros(n_feature) #jitne number of columns utne feautres to utna array size chaiye humko
        threshold=1e-4
        costs=[]

        for i in range(self.iterations):
            z=np.dot(X,self.weights) + self.bias
            y_pred=self.sigmoid_probability(z)
            cost = self.compute_cost(y, y_pred)
            costs.append(cost)
            gradient_weight = (1 / n_sample) * np.dot(X.T, (y_pred - y))
            gradient_bias = (1 / n_sample) * np.sum(y_pred - y)


            step_size_weight=self.alpha*gradient_weight
            step_size_bias=self.alpha*gradient_bias

            if np.all(step_size_weight) < threshold and np.abs(step_size_bias) < threshold:
                print(f"Converged after {i} iterations")
                break

            self.weights=self.weights-step_size_weight
            self.bias=self.bias-step_size_bias

            if i % 100==0 or i==self.iterations-1:
                cost=self.compute_cost(y,y_pred)
                print(f"Iteration {i}, Cost: {cost}")

    def predict_values(self,X):
        z=np.dot(X, self.weights) + self.bias
        return self.sigmoid_probability(z) 
    
    def predict_binary(self, X):
        probabilities=self.predict_values(X)  
        return (probabilities >= 0.5).astype(int) 
    
    def model_accuracy(self, X, y):
        y_pred=self.predict_binary(X) 
        correct_predictions=np.sum(y_pred == y) 
        return (correct_predictions/len(y)) * 100

    def plot_comparison(self, X, y):
        y_pred = self.predict_binary(X)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(y)), y, label='Actual Values', marker='o', linestyle='--', color='blue')
        plt.plot(range(len(y_pred)), y_pred, label='Predicted Values', marker='x', linestyle='-', color='red')
        plt.xlabel('Sample Index')
        plt.ylabel('Class Label')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sigmoid_curve_probability(self,X,y):
        z=np.dot(X, self.weights) + self.bias
        sigmoid_value=self.sigmoid_probability(z)
        plt.plot(z,sigmoid_value)
        plt.plot(z,sigmoid_value,color='green',label='Sigmoid Curve',linewidth=2)
        plt.scatter(z,sigmoid_value, color='red', label='Predicted Values', zorder=5)
        plt.axvline(0,color='black', linewidth=1)# Line at z = 0
        plt.axhline(0.5, color='black', linewidth=1)# Line at y = 0.5
        plt.xlabel('Z Value (Input to Sigmoid)')
        plt.ylabel('Sigmoid Output')
        plt.title('Sigmoid Curve with Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        


