import numpy as np

# this is a general purpose function so no need to include it in the class of Kmeans although you can.
def euclidean_distance(point_1,point_2):
    distance=np.linalg.norm(point_1-point_2)
    return distance

class KMeansplusplus:
    def __init__(self,K,iterations,tolerance):
        self.K=K                        # number of clusters
        self.max_iterations=iterations  # number of times the for loop will run
        self.tolerance=tolerance        # minimum allowed change in centroids, suppose if centroid is 2.05 and after another iter it is 2.5001 we stop
        self.n_samples=None
        self.n_features=None
        self.cost_history=[]
    
    def centroid_initlization(self,X):
        np.random.seed(42)
        centroids=[]
        # pick the first centorid randomly
        first_index=np.random.choice(len(X))
        centroids.append(X[first_index])
        for i in range(1,self.K):
            distance=[]
            for x in X:
                dist=min([euclidean_distance(x,c)**2 for c in centroids])
                distance.append(dist)

            distance=np.array(distance)
            probability=distance/distance.sum()

            next_index=np.random.choice(len(X),p=probability) # using random choice becuase if not then the algo would awlwasy pick outliers
            centroids.append(X[next_index])
        return np.array(centroids)


    def fit(self,X):
        self.n_samples,self.n_features=X.shape
        self.centroids=self.centroid_initlization(X) # random initilization of centroid points

        for i in range(self.max_iterations):
            self.clusters=self.create_cluster(X) # initial assingment with random centroids
            old_centroids=self.centroids.copy()
            self.centroids = np.array([X[cluster].mean(axis=0) if len(cluster) > 0 else old_centroids[i]
                                       for i, cluster in enumerate(self.clusters)])
            
            cost = self.compute_cost(X)
            self.cost_history.append(cost)
            
            difference=np.linalg.norm(self.centroids-old_centroids)
            if difference < self.tolerance:
                break
        
        self.labels_ = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                self.labels_[sample_idx] = cluster_idx


    def create_cluster(self,X):
        clusters=[[] for i in range(self.K)] # creates a nested list [[for cluster 0] [cluster 1]....]
        for index,value in enumerate(X):
            centroid_idx=self.nearest_centriod(value) # returns the index of the closest centroid
            clusters[centroid_idx].append(index)
        
        return clusters
    
    def nearest_centriod(self,data_point):
        distances=[euclidean_distance(data_point,centroid) for centroid in self.centroids]
        return np.argmin(distances)

    def predict(self,X):
        predictions = [self.nearest_centriod(x) for x in X]
        return np.array(predictions)
    
    def compute_cost(self,X):
        cost=0
        for cluster_index,value in enumerate(self.clusters):
            centroid=self.centroids[cluster_index]
            for point_idx in value:
                cost += euclidean_distance(X[point_idx], centroid) ** 2
        return cost


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.6, random_state=0)

model = KMeansplusplus(K=5,iterations=100,tolerance=0.001)
model.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis')
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], s=200, c='red', marker='X')
plt.title("K-Means Clustering (Custom)")
plt.show()

plt.plot(model.cost_history)
plt.title("Cost Function Decrease Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost (Inertia)")
plt.grid(True)
plt.show()
