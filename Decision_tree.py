import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class Node:
    """
   Represent Node of a Decision tree(Whether it is leaf node or Decision Node)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, 
                 value=None, samples=None, gini=None):
        """
        Parameters:-
        feature : Index of feature for spliting(int)
           
        threshold : Threshold value for spliting(float value)
        left : left child node
        right : right child node
        value : predicted value of leaf node(int/float) 
        samples : number of samples in the node(int value)
        gini : Impurity of the Node(float value)
        """
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold for split
        self.left = left           # Left child
        self.right = right         # Right child
        self.value = value         # Prediction value (leaf nodes)
        self.samples = samples     # Number of samples
        self.gini = gini          # Gini impurity
    
    def is_leaf(self):
        """Check if node is a leaf node"""
        return self.value is not  None


class DecisionTree:
    """
   Decision Tree Classifier (building binary tree recursively)
    """ 
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini'):
        """
        Initialization of Decision Tree
        
        Parameters:-
        
        max_depth :  Maximum depth of the tree (default: 10)
         min_samples_split :min samples required for split(default value 2)
        min_samples_leaf : min samples in leaf node(default value=1)
        criterion : Split criteria("Gini" or "Entropy")
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_features = None
        self.n_classes = None
    
    def fit(self, X, y):
        """
        Build the decision tree
        
        Parameters:-
       
        X : Training data(ndarray of shape(n_samples,n_features)) 
        y : Target values of the model(ndarray(n_samples))
        """
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        print(f"Building decision tree...")
        print(f"  Features: {self.n_features}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Min samples split: {self.min_samples_split}")
        
        # Build tree recursively
        self.root = self._build_tree(X, y, depth=0)
        print(f" Tree built successfully!")
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree
        
        Returns:
        --------
        Node : A decision node or leaf node
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Calculate Gini impurity for this node based on Target labels
        gini = self._gini_impurity(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = self._most_common_label(y) # leaf node created
            return Node(value=leaf_value, samples=n_samples, gini=gini)
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, samples=n_samples, gini=gini)
        
        # Split features
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        if (np.sum(left_indices) < self.min_samples_leaf or 
            np.sum(right_indices) < self.min_samples_leaf):# check min_samples constraint
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value, samples=n_samples, gini=gini)
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        # Decision node creation:-
        return Node(feature=best_feature, threshold=best_threshold,
                   left=left_child, right=right_child,
                   samples=n_samples, gini=gini)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on
        
        Returns:-
        best fetures:- Index of best Features(int)
        best threshold:- best Threshold Value(float)
        """
       
    
       
        best_gain = -1
        best_feature = None
        best_threshold = None
        for feature in range(self.n_features):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature])
            
            # Looping in each Threshold
            for threshold in thresholds:
             
                gain = self._information_gain(X, y, feature, threshold)
                
                # Update best split if this split is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X, y, feature, threshold):
        """
        Calculate information gain from a split
        
        Information Gain = Parent Impurity - Weighted Average of Children Impurity
        """
        # Parent impurity
        parent_impurity = self._calculate_impurity(y)
        
        # Split data
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        
        # If split doesn't divide the data, no gain i.e it is a leaf Node
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0
        
        # Calculate weighted impurity of children
        n = len(y)
        n_left, n_right = np.sum(left_indices), np.sum(right_indices)
        
        left_impurity = self._calculate_impurity(y[left_indices])
        right_impurity = self._calculate_impurity(y[right_indices])
        
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        # Information gain
        gain = parent_impurity - weighted_impurity
        return gain
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on chosen criterion"""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity
        
        Gini = 1 - Σ(pᵢ²)
        where pᵢ is the probability of class i
        """
        if len(y) == 0:
            return 0
        
        # Count classes
        counter = Counter(y)
        impurity = 1.0
        
        # Calculate Gini
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob ** 2
        
        return impurity
    
    def _entropy(self, y):
        """
        Calculate entropy
        
        Entropy = -Σ(pᵢ * log₂(pᵢ))
        """
        if len(y) == 0:
            return 0
        
        counter = Counter(y)
        entropy = 0.0
        
        for count in counter.values():
            prob = count / len(y)
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _most_common_label(self, y):
        """Return the most common label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """
        Make predictions for input data
        
        Parameters:
    
        X : numpy array of shape (n_samples, n_features)
            Data to predict
            
        Returns:
        
        predictions : numpy array
            Predicted labels
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse tree to make prediction for a single sample
        """
        # If leaf node, return value
        if node.is_leaf():
            return node.value
        
        # Decide which child to follow
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """
        Print tree structure in readable format
        """
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print(f"{'  ' * depth}{prefix}Predict: {node.value} (samples={node.samples}, gini={node.gini:.3f})")
        else:
            print(f"{'  ' * depth}{prefix}X[{node.feature}] <= {node.threshold:.3f} (samples={node.samples}, gini={node.gini:.3f})")
            self.print_tree(node.left, depth + 1, "Left: ")
            self.print_tree(node.right, depth + 1, "Right: ")
    
    def get_depth(self, node=None):
        """Calculate actual depth of the tree"""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 0
        
        left_depth = self.get_depth(node.left)
        right_depth = self.get_depth(node.right)
        
        return max(left_depth, right_depth) + 1
    
    def count_nodes(self, node=None):
        """Count total nodes in the tree"""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 1
        
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)



# HELPER FUNCTIONS:-

def generate_classification_data(n_samples=200):
    """Generate sample data for classification"""
    np.random.seed(42)
    
    # Create non-linear decision boundary
    X = np.random.randn(n_samples, 2)
    
    # Complex decision rule
    y = np.zeros(n_samples)
    y[(X[:, 0] > 0) & (X[:, 1] > 0)] = 1
    y[(X[:, 0] < 0) & (X[:, 1] < 0)] = 1
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    return X, y.astype(int)


def plot_decision_boundary(X, y, model, title="Decision Tree"):
    """Visualize decision boundary"""
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Decision boundary
    plt.subplot(1, 3, 1)
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', 
                edgecolors='k', s=50, alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', 
                edgecolors='k', s=50, alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of max_depth
    plt.subplot(1, 3, 2)
    depths = [1, 3, 5, 10]
    train_accs = []
    
    for depth in depths:
        temp_model = DecisionTree(max_depth=depth)
        temp_model.fit(X, y)
        acc = temp_model.score(X, y)
        train_accs.append(acc)
    
    plt.plot(depths, train_accs, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Max Depth')
    plt.ylabel('Training Accuracy')
    plt.title('Effect of Tree Depth')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Tree complexity
    plt.subplot(1, 3, 3)
    node_counts = []
    
    for depth in depths:
        temp_model = DecisionTree(max_depth=depth)
        temp_model.fit(X, y)
        nodes = temp_model.count_nodes()
        node_counts.append(nodes)
    
    plt.bar(range(len(depths)), node_counts, tick_label=depths, 
            color='skyblue', edgecolor='black')
    plt.xlabel('Max Depth')
    plt.ylabel('Number of Nodes')
    plt.title('Tree Complexity')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
# MAIN EXECUTION

if __name__ == "__main__":
    print("=" * 70)
    print("DECISION TREE FROM SCRATCH")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating classification data...")
    X, y = generate_classification_data(n_samples=200)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    print("\n2. Training Decision Tree...")
    dt = DecisionTree(max_depth=5, min_samples_split=2, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    
    # Tree information
    print(f"\n3. Tree Information:")
    print(f"   Actual depth: {dt.get_depth()}")
    print(f"   Total nodes: {dt.count_nodes()}")
    print(f"   Leaf nodes: ~{dt.count_nodes() // 2}")
    
    # Make predictions
    print("\n4. Making predictions...")
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    
    print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Print tree structure
    print("\n5. Tree Structure (first 3 levels):")
    print("-" * 70)
    # Limit depth for readability
    temp_tree = DecisionTree(max_depth=3)
    temp_tree.fit(X_train, y_train)
    temp_tree.print_tree()
    print("-" * 70)
    
    # Sample prediction with explanation
    print("\n6. Sample Prediction Walkthrough:")
    sample = X_test[0]
    print(f"   Input: {sample}")
    print(f"   Actual label: {y_test[0]}")
    print(f"   Predicted label: {dt.predict([sample])[0]}")
    
    # Visualize
    print("\n7. Visualizing decision boundary...")
    plot_decision_boundary(X_train, y_train, dt, 
                          title=f"Decision Tree (depth={dt.max_depth})")
    
 
    

    
