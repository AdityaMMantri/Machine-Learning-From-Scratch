import numpy as np
from graphviz import Digraph
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class Node:
    def __init__(self,feature=None,left_child=None,right_child=None,value=None,threshold=None):
        self.feature=feature
        self.left_child=left_child
        self.right_child=right_child
        self.value=value
        self.threshold=threshold

    def leaf_node(self):
        if self.value is not None:
            return True
        else:
            return False
# feature var-> tells the index of the feature that is vo feature jiske uppar node bana hai
# left_child-> us node ka left sub part
# right_child-> us node ka right sub part
# thershold-> it is that value of the feature on which we have divided the dataset, basically vo value jispe max IG mil raha hai
# value-> if the node is the leaf node then it will have a value that is the class label

class Decision_tree:
    def __init__(self,max_depth,min_sample_split):
        self.max_depth=max_depth
        self.min_sample_split=min_sample_split
        self.root=None

    def fit(self, X, y):
        features = list(range(X.shape[1]))
        self.root = self.decision_tree(X, y, features)

    def decision_tree(self,X,y,features,depth=0):
        number_of_sample,number_of_features=X.shape
        number_labels=len(np.unique(y))

        if depth >= self.max_depth or number_of_sample <= self.min_sample_split or number_labels == 1 or len(features) == 0:
            return Node(value=self.most_common_label(y))

        best_feature, best_threshold, left_indices, right_indices = self.best_split(X, y, features)

        if best_feature is None:
            return Node(value=self.most_common_label(y))

        features_left = features.copy()
        features_left.remove(best_feature)

        left_child = self.decision_tree(X[left_indices], y[left_indices], features_left, depth + 1)
        right_child = self.decision_tree(X[right_indices], y[right_indices], features_left, depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left_child=left_child, right_child=right_child)


    def most_common_label(self, y):
        # counts the number of class labels in the output variable and returns the most
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)
        return unique_labels[most_common_index]
    
    def entropy(self,y):
        labels,counts=np.unique(y,return_counts=True)
        probability=counts/np.sum(counts)
        entropy = -np.sum(probability * np.log2(probability + 1e-9))
        return entropy
    
    def information_gain(self,parent,left_child,right_child):
        weighted_left=len(left_child)/len(parent)
        weighted_right=len(right_child)/len(parent)
        entropy_parent=self.entropy(parent)
        weighted_child_entropy = (weighted_left*self.entropy(left_child)) + (weighted_right*self.entropy(right_child)) 
        return entropy_parent-weighted_child_entropy
    
    def best_split(self,X,y,features):
        best_gain=-1
# We want to maximize this value — it keeps track of the highest Information Gain (IG) found so far
        best_threshold=None
# optimal threshold value at which IG is maximum for a feature
        best_feature=None
# returns the index of the feature with the max IG
        best_left_child_index=None
# A boolean array or index list: True (or selected index) if a sample goes to the left child (feature <= threshold)
        best_right_child_index=None
# A boolean array or index list: True (or selected index) if a sample goes to the right child (feature > threshold)

        n_samples, n_features= X.shape
        for feature_index in features:
            feature_value=X[:,feature_index]
            All_thresholds=np.unique(feature_value)

            for threshold in All_thresholds:
                left_indices = feature_value<= threshold
                right_indices = feature_value> threshold

                if np.sum(left_indices)==0 or np.sum(right_indices)==0:
                    continue

                IG=self.information_gain(y,y[left_indices],y[right_indices])

                if IG>best_gain:
                    best_gain=IG
                    best_feature=feature_index
                    best_left_child_index=left_indices
                    best_right_child_index=right_indices
                    best_threshold=threshold

        return best_feature,best_threshold,best_left_child_index,best_right_child_index
    
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample, node):
        if node.leaf_node():
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left_child)
        else:
            return self._predict_sample(sample, node.right_child)
        
def visualize_tree_graphviz(root_node):
    dot = Digraph()

    def add_nodes_edges(node, parent_id=None, edge_label=""):
        node_id = str(id(node))  # unique node id

        # Node label
        if node.leaf_node():
            label = f"Predict: {node.value}"
        else:
            label = f"X[{node.feature}] <= {node.threshold:.2f}"

        dot.node(node_id, label)

        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)

        if not node.leaf_node():
            add_nodes_edges(node.left_child, node_id, "True")
            add_nodes_edges(node.right_child, node_id, "False")

    add_nodes_edges(root_node)
    return dot
    
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

tree = Decision_tree(max_depth=4,min_sample_split=2)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
dot = visualize_tree_graphviz(tree.root)
dot.render("final_tree_visualization_id3", format="png", cleanup=True)
