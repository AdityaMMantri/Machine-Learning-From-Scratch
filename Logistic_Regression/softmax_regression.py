import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,classification_report

class SoftmaxRegression:
    def __init__(self,init="random",learning_rate=0.01,epochs=100,): # default values provided
        self.lr=learning_rate
        self.epochs=epochs
        self.init=init
        self.W=None
        self.bias=None

        #meta data
        self.n_classes=None
        self.in_features=None

    def correct_probabilities(self,y,y_pred):
        '''
        basically we are doing indexing because cross entropy only requries the correct class probablities.

        y_pred : ndarray of shape (n_samples, n_classes)

            Softmax output probabilities.
            Example:
            [[0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5]]

        y : ndarray of shape (n_samples,)

            True class labels.
            Example:
            [0, 1, 2]

        correct_probs : ndarray of shape (n_samples,)
            y_pred[
                [0,0]
                [1,1]
                [2,2]
                ]
            Probabilities assigned to the true classes.
            Example:
            [0.7, 0.8, 0.5]
        '''
        n_sample=y.shape[0]
        correct_probs=y_pred[np.arange(n_sample),y]
        return correct_probs
    
    def cross_entropy_loss(self,y,y_pred):
        correct_probs=self.correct_probabilities(y,y_pred)
        # cross entropy formula is -(summation(log(y_pred)))/N
        loss=-np.mean(np.log(correct_probs+1e-9))
        return loss
    
    def softmax(self,logits):
        '''
        subtract max logit from each row
        prevents:
        np.exp(1000) -> overflow
        This does NOT change probabilities because
        softmax depends only on relative differences.

        axis = 1 because we want row operation as each row is a sample
        '''
        logits=logits-np.max(logits,axis=1,keepdims=True)
        exp_logits=np.exp(logits)
        probabilities=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        return probabilities
    
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.n_classes=len(np.unique(y))
        self.in_features=n_features

        if self.init=="zero":
            self.W=np.zeros((self.in_features,self.n_classes)) # zero initlization of weight matrix
        elif self.init=="random":
            self.W=np.random.randn(self.in_features,self.n_classes)*0.01    # 0.01 beacuse wieghts might be very larges so 
                                                                            # softmax becomes nearly one-hot immediately.Gradients become unstable/useless.
        elif self.init=="xavier":
            std=np.sqrt(1/self.in_features)
            self.W=np.random.randn(self.in_features,self.n_classes)*std
        elif self.init=="he":
            std=np.sqrt(2/self.in_features)
            self.W=np.random.randn(self.in_features,self.n_classes)*std
        else:
            raise ValueError("Unsupported Initilzation type")
        
        self.bias=np.zeros((1,self.n_classes))

        for epoch in range(self.epochs):
            logits=X @ self.W + self.bias
            probabilities=self.softmax(logits=logits)
            loss=self.cross_entropy_loss(y=y,y_pred=probabilities)
            # the gradeint of the cross entropy loss with softmax resolves to y_pred/probabilities - y_actual
            # now instead of making y_actual as one hot vector what we do subtract 1 from the correct probability
            # basically suppose y_pred/probabilities = [0.1,0.7,0.2] and the correct class is 0 the model has 
            # given 0.1 score to it the one hot vector would be [1,0,0] so actual output gradient would be -> [-0.9,0.7,0.2]
            # but look the places where it is 0 we dont need to store it or convert it to one hot vector, we can have a
            # similar approch to that of correct_probabilities function where we form the indexing as 
            # y_pred[(np.arange(n_samples)),y]-=1 this will form the gradient
            dZ=probabilities.copy()
            dZ[np.arange(n_samples),y]-=1
            dZ/=n_samples # Average gradient

            dW=X.T @ dZ
            db=np.sum(dZ,axis=0,keepdims=True)

            # parameter update
            self.W-=self.lr*dW
            self.bias-=self.lr*db

        return self
    
    def predict_probabilities(self,X):
        logits=X @ self.W + self.bias
        probabilities=self.softmax(logits=logits)
        return probabilities
    
    def predict(self,X):
        probabilities=self.predict_probabilities(X)
        y_pred=np.argmax(probabilities,axis=1) # asumming the y variable is encoded so argmax will return the index of max probab.
        return y_pred
    
X,y=make_classification(n_samples=5000,         # number of data points
                        n_features=10,          # number of total features
                        n_informative=6,        # number of total useful/distinct features
                        n_redundant=4,          # Creates features that are linear combinations of informative features. 
                        n_classes=5,            # Number of output classes.
                        n_clusters_per_class=1, #
                        random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

my_model=SoftmaxRegression(init="random",learning_rate=0.1,epochs=1000)
my_model.fit(X=X_train,y=y_train)
y_pred_custom=my_model.predict(X=X_test)
print("\n==============================")
print("CUSTOM MODEL METRICS")
print("==============================")

custom_accuracy = accuracy_score(
    y_test,
    y_pred_custom
)

custom_f1 = f1_score(
    y_test,
    y_pred_custom,
    average="weighted"
)

custom_recall = recall_score(
    y_test,
    y_pred_custom,
    average="weighted"
)

custom_conf_matrix = confusion_matrix(
    y_test,
    y_pred_custom
)

print(f"Accuracy : {custom_accuracy:.4f}")
print(f"F1 Score : {custom_f1:.4f}")
print(f"Recall   : {custom_recall:.4f}")

print("\nConfusion Matrix:")
print(custom_conf_matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_custom
    )
)

sk_model=LogisticRegression(multi_class="multinomial",max_iter=500,solver="saga")
sk_model.fit(X=X_train,y=y_train)
y_pred_sk=sk_model.predict(X=X_test)
print("\n==============================")
print("SCIKIT LEARN METRICS")
print("==============================")

sk_accuracy = accuracy_score(
    y_test,
    y_pred_sk
)

sk_f1 = f1_score(
    y_test,
    y_pred_sk,
    average="weighted"
)

sk_recall = recall_score(
    y_test,
    y_pred_sk,
    average="weighted"
)

sk_conf_matrix = confusion_matrix(
    y_test,
    y_pred_sk
)

print(f"Accuracy : {sk_accuracy:.4f}")
print(f"F1 Score : {sk_f1:.4f}")
print(f"Recall   : {sk_recall:.4f}")

print("\nConfusion Matrix:")
print(sk_conf_matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_sk
    )
)

print("\n==============================")
print("FIRST 10 PREDICTIONS")
print("==============================")

print("True Labels:")
print(y_test[:10])

print("\nYour Model:")
print(y_pred_custom[:10])

print("\nScikit Learn:")
print(y_pred_sk[:10])