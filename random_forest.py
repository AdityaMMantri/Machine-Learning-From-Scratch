from CART_RANDOM import Decision_tree
import numpy as np


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