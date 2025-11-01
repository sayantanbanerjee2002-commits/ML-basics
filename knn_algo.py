import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNearestNeighbors:
    """
    K-Nearest Neighbors implemented from scratch( Supports both classification and regression)
    """
    
    def __init__(self, k=3, distance_metric='euclidean', task='classification'):
        """
        Initialize the KNN model
        Parameters:-
        k : number of neighbours to be considered(Hyperparameter,default value = 3) 
        distance_metric : Euclidean distance(string) 
        task : data type is String(default = Classification)
        """
        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store the training data  
        Parameters:-
        X : Training data(ndarray(n_samples, n_features))
        y : Target values(ndarray of shape(n_samples))
        """
        self.X_train = X
        self.y_train = y
        print(f"Stored {len(X)} training samples")
        print(f"Number of features: {X.shape[1]}")
        print(f"Task: {self.task}")
    
    def euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points:
        """
        return np.sqrt  ( np.sum ((x1 - x2)**2))
    
    def manhattan_distance(self, x1, x2):
        """
        Calculate Manhattan distance between two points:
        """
        return np.sum(np.abs(x1 - x2))
    
    
    def calculate_distance(self, x1, x2):
        """
        Calculate distance based on chosen metric
        """
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict(self, X):
        """
        Make predictions for input data:-
        Parameters:
        X : Data to make prediction (ndarray) 
        Returns:
        predictions : predicted value(ndarray of shape(x_samples))
        """
        predictions = [self._predict_single(x) for x in X] # List comprehension
        return np.array(predictions)
    
    def _predict_single(self, x):
        """
        Make prediction for a single data point
        
        Steps:
        1. Calculate distances to all training points
        2. Find K nearest neighbors
        3. Majority Voting (classification) or average (regression)
        """
        # Calculate distances to all training points
        distances = [self.calculate_distance(x, x_train) 
                    for x_train in self.X_train] # List Comprehension
        
        # Get indices of K nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of K nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Make prediction based on task
        if self.task == 'classification':
            #  Apply Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        else:
            # Apply for Regression(Average of labels)
            return np.mean(k_nearest_labels)
    
    def predict_with_distances(self, x):
        """
        Make prediction and return neighbor information
      Returns:-
        prediction, k_indices, k_distances
        """
        # Calculate distances to all training points
        distances = [self.calculate_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get indices of K nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_distances = [distances[i] for i in k_indices] # List Comprehension
        
        # Get labels of K nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Make prediction
        if self.task == 'classification':
            most_common = Counter(k_nearest_labels).most_common(1)
            prediction = most_common[0][0]
        else:
            prediction = np.mean(k_nearest_labels)
        
        return prediction, k_indices, k_distances
    
    def score(self, X, y):
        """
        Calculate accuracy (classification) or R² score (regression)
      Parameters:-
        X :Input Features(ndarray)
        y :True values(ndarray)
     Returns:-
       score: Accuracy score 
       
        """
        predictions = self.predict(X) # predictions store the predicted value of the model tarined by datasets
        
        if self.task == 'classification':
            # Accuracy
            return np.mean(predictions == y)
        else:
            # R² score
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - predictions) ** 2)
            return 1 - (ss_residual / ss_total)

# HELPER FUNCTIONS:-

def generate_classification_data(n_samples=150):
    """Generate sample data for classification"""
    np.random.seed(42)
    
    # Three classes in different regions
    # Class 0: bottom-left
    X_class0 = np.random.randn(n_samples//3, 2) * 0.6 + np.array([-2, -2])
    y_class0 = np.zeros(n_samples//3)
    
    # Class 1: top-right
    X_class1 = np.random.randn(n_samples//3, 2) * 0.6 + np.array([2, 2])
    y_class1 = np.ones(n_samples//3)
    
    # Class 2: top-left
    X_class2 = np.random.randn(n_samples//3, 2) * 0.6 + np.array([-2, 2])
    y_class2 = np.full(n_samples//3, 2)
    
    # Combine
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.hstack([y_class0, y_class1, y_class2])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def generate_regression_data(n_samples=100):
    """Generate sample data for regression"""
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, n_samples)
    return X, y


def plot_classification_results(X, y, model, test_point=None):
    """Visualize classification results"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Decision boundary
    plt.subplot(1, 3, 1)
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    colors = ['blue', 'red', 'green']
    for i in range(int(y.max()) + 1):
        plt.scatter(X[y==i, 0], X[y==i, 1], 
                   c=colors[i], label=f'Class {i}', 
                   edgecolors='k', s=50, alpha=0.7)
    
    if test_point is not None:
        plt.scatter(test_point[0], test_point[1], 
                   c='yellow', s=200, marker='*', 
                   edgecolors='black', linewidths=2,
                   label='Test Point')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'KNN Decision Boundary (K={model.k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of K on decision boundary
    plt.subplot(1, 3, 2)
    k_values = [1, 5, 15]
    
    for idx, k_val in enumerate(k_values):
        temp_model = KNearestNeighbors(k=k_val, task='classification')
        temp_model.fit(X, y)
        
        Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.subplot(3, 3, 4 + idx)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        for i in range(int(y.max()) + 1):
            plt.scatter(X[y==i, 0], X[y==i, 1], 
                       c=colors[i], s=20, alpha=0.6, edgecolors='k')
        plt.title(f'K={k_val}')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Distance metrics comparison
    plt.subplot(1, 3, 3)
    metrics = ['euclidean', 'manhattan']
    accuracies = []
    
    for metric in metrics:
        temp_model = KNearestNeighbors(k=5, distance_metric=metric, task='classification')
        temp_model.fit(X, y)
        acc = temp_model.score(X, y)
        accuracies.append(acc)
    
    plt.bar(metrics, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Distance Metric Comparison')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()


def plot_regression_results(X, y, model):
    """Visualize regression results"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Predictions
    plt.subplot(1, 3, 1)
    X_test = np.linspace(0, 5, 300).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    plt.scatter(X, y, c='blue', label='Training data', alpha=0.6)
    plt.plot(X_test, y_pred, c='red', linewidth=2, label=f'KNN (K={model.k})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'KNN Regression (K={model.k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of K
    plt.subplot(1, 3, 2)
    k_values = [1, 5, 15, 30]
    
    plt.scatter(X, y, c='blue', label='Training data', alpha=0.4, s=30)
    
    for k_val in k_values:
        temp_model = KNearestNeighbors(k=k_val, task='regression')
        temp_model.fit(X, y)
        y_pred = temp_model.predict(X_test)
        plt.plot(X_test, y_pred, linewidth=2, label=f'K={k_val}')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Effect of K on Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: R² score vs K
    plt.subplot(1, 3, 3)
    k_range = range(1, 31)
    scores = []
    
    for k_val in k_range:
        temp_model = KNearestNeighbors(k=k_val, task='regression')
        temp_model.fit(X, y)
        score = temp_model.score(X, y)
        scores.append(score)
    
    plt.plot(k_range, scores, marker='o', linewidth=2, markersize=5)
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs K')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Execution of the Programm

if __name__ == "__main__":
    print("=" * 60)
    print("K-NEAREST NEIGHBORS - CLASSIFICATION")
    print("=" * 60)
    
    # Generate classification data
    print("\n1. Generating classification data...")
    X_class, y_class = generate_classification_data(n_samples=150)
    print(f"   Data shape: X={X_class.shape}, y={y_class.shape}")
    print(f"   Classes: {np.unique(y_class)}")
    
    # Split data
    split_idx = int(0.8 * len(X_class))
    X_train, X_test = X_class[:split_idx], X_class[split_idx:]
    y_train, y_test = y_class[:split_idx], y_class[split_idx:]
    
    # Create and train model
    print("\n2. Training KNN Classifier...")
    knn_clf = KNearestNeighbors(k=5, distance_metric='euclidean', task='classification')
    knn_clf.fit(X_train, y_train)
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = knn_clf.predict(X_test)
    accuracy = knn_clf.score(X_test, y_test)
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed prediction example
    print("\n4. Detailed Prediction Example:")
    test_point = X_test[0]
    pred, neighbors, distances = knn_clf.predict_with_distances(test_point)
    print(f"   Test point: {test_point}")
    print(f"   Prediction: Class {int(pred)}")
    print(f"   Neighbor distances: {[f'{d:.3f}' for d in distances]}")
    print(f"   Neighbor labels: {y_train[neighbors]}")
    
    # Visualize
    print("\n5. Visualizing classification results...")
    plot_classification_results(X_train, y_train, knn_clf, test_point)
    
    # REGRESSION :-
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS - REGRESSION")
    print("=" * 60)
    
    print("\n1. Generating regression data...")
    X_reg, y_reg = generate_regression_data(n_samples=100)
    print(f"   Data shape: X={X_reg.shape}, y={y_reg.shape}")
    
    print("\n2. Training KNN Regressor...")
    knn_reg = KNearestNeighbors(k=5, task='regression')
    knn_reg.fit(X_reg, y_reg)
    
    print("\n3. Making predictions...")
    r2_score = knn_reg.score(X_reg, y_reg)
    print(f"   R² Score: {r2_score:.4f}")
    
    print("\n4. Visualizing regression results...")
    plot_regression_results(X_reg, y_reg, knn_reg)