import numpy as np

class DBSCAN:
    def __init__(self,epsilon,minpts,metric=None):
        self.minpts=minpts
        self.epsilon=epsilon
        self.distance_metric=metric
        self.labels=None
    
    def neighbour_points(self,data_point1,data):
        neighbours=set()
        for data_point in range(len(data)):
            if self.distance(data[data_point1],data[data_point])<=self.epsilon:
                neighbours.add(data_point)
        return list(neighbours)
    
    def distance(self, point1, point2):
        try:
            if self.distance_metric=="euclidean":
                distance=np.sqrt(np.sum((point1 - point2) **2))
            elif self.distance_metric=="manhattan":
                distance=np.sum(np.abs(point1 - point2))
            elif self.distance_metric=="cosine":
                norm1=np.linalg.norm(point1)
                norm2=np.linalg.norm(point2)
                if norm1 == 0 or norm2 == 0:
                    raise ValueError("Zero vector found cosine distance is undefined.")
                distance = 1 - np.dot(point1, point2) / (norm1 * norm2)
            else:
                raise ValueError(f"Invalid distance metric: '{self.distance_metric}'")
            return distance
        except Exception as e:
            print(f"Error computing distance between {point1} and {point2}: {e}")
            return None
        
    def core_points(self,data):
        core_points=[]
        for i in range(0,len(data)):
            neighbours=self.neighbour_points(i,data)
            if len(neighbours)>=self.minpts:
                core_points.append(i)
            
        return core_points
    
    def fit(self,data):
        labels=[0] * len(data)
        cluster_id = 0 # will increment after a new cluster is found 
        for point in range(len(data)):
            if labels[point] != 0: # means the point has been visited
                continue
            neighbours=self.neighbour_points(point,data) # neighbours of the current data point being traversed

            if len(neighbours)<self.minpts:# the current point being traversed has less than the min pts in its neighbours label it as noise point
                labels[point] = -1 
            else: # the point is a core point
                cluster_id+=1
                labels=self.grow_cluster(point,data,cluster_id,labels)
        self.labels=labels
        return labels
        
    def grow_cluster(self,point_id,data,cluster_id,labels):
        neighbours_of_current_point=self.neighbour_points(point_id,data) # a core point is passed from fit so we fetch its neighbours
        labels[point_id]=cluster_id # assingn the core point a id
        i=0
        seen=set(neighbours_of_current_point)
        while i<len(neighbours_of_current_point):
            neighbour=neighbours_of_current_point[i]
            if labels[neighbour]==-1: # previously labelled noise point is reachable now
                labels[neighbour]=cluster_id
            elif labels[neighbour]==0: # previosuly univisted point is reachable now
                labels[neighbour]=cluster_id
            new_neighbours=self.neighbour_points(neighbour,data)
            if len(new_neighbours)>=self.minpts:
                for new_n in new_neighbours:
                        if new_n not in seen:
                            neighbours_of_current_point.append(new_n)
                            seen.add(new_n) # agar naya point bhi core hai to uske neighours bhi added

            i+=1
        return labels
                



from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_moons(n_samples=500, noise=0.01)

# Run DBSCAN
model = DBSCAN(epsilon=0.2, minpts=5, metric="euclidean")
labels = model.fit(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')
plt.title("DBSCAN Clustering")
plt.show()





# 0-> means univisted 
# -1-> noise points
# postive integers-> cluster id's



        





