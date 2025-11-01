import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic Regression implemented from scratch using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Logistic Regression model
         Parameters:-
         
        learning_rate : Step Size of Gradient Descent(float valye), by default value = 0.1
        iterations : Number of iteration to run Gradient descent(int value, default value 1000)
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []  # Track cost over iterations
    
    def sigmoid(self, z):
        """
        Sigmoid activation function that conveerts any value in the range[0,1]
      Parameters:
        z : taking Input as Linear Combination
     Returns:-
         sigmoid :ndarray(output between 0 and 1)
        """
        # Clip values to prevent overflow in exp
        z = np.clip(z,-500,500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the model using gradient descent
        Parameters:-
        X : Training Data(ndarray(n_samples, n_features))
        y : Target Value(ndarray of shape(n_samples))
        """
        # Get number of samples and features
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zero
        self.weights = np.ones(n_features)
        self.bias = 1.5
        
        # implement Gradient Descent
        for i in range(self.iterations):
            # Linear combination: z = wx + b <=> y = mx + c
            linear_model = X @ self.weights + self.bias
            
            # Apply sigmoid function to get predictions
            y_predicted = self.sigmoid(linear_model)
            
            # Calculate gradients:- Apply in sigmoid output
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias:- Convergence Algorithm
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate and store cost for this iteration
            cost = self._calculate_cost(y, y_predicted) # Here y:- Actual output,y_predicted:- Predicted output
            self.cost_history.append(cost)
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.4f}")
    
    def predict_proba(self, X):
        """
        Predict probability estimates
         Parameters:-
        
        X : Data to make Predictions(ndarray)
      Returns:-
        probabilities : Return ndarray of shape(n_shape):- belongs to[0,1]
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return y_predicted
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
      Parameters:-
        X :  Data to make Predictions on threshold( default threshold value = 0.05)
      Returns:-
        predictions : Predicted Class labels(0 or 1)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int) # astype(int) converts boolean value into integers
        return predictions
    
    def _calculate_cost(self, y_true, y_pred):
        """
        Calculate Binary  cost
     Parameters:-
        y_true : Actual output of ndarray
        y_pred : Predicted Probabilites of ndarray
     Returns:-
        cost : Binary Cost Function
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # Here Min_value = epsilon, Max_value = 1-epsilon
        
        n_samples = len(y_true)
        cost = -(1/n_samples) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return cost
    
    def score(self, X, y):
        """
        Calculate accuracy score(R**2)
     Parameters:-
        X : Input Features(ndarray)
        y :True Labels(ndarray)
     Returns:-
        accuracy : Accuracy Score(Float)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

# HELPER FUNCTIONS FOR EVALUATION

def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 score"""
    #  Define True Positives, False Positives, False Negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    }

# Testing:-

def generate_sample_data(n_samples=200):
    """Generate sample binary classification data"""
    np.random.seed(52)
    
    # Class 0: centered at (-2, -2)
    X_class0 = np.random.randn(n_samples//2, 2) + np.array([-2, -2])
    y_class0 = np.zeros(n_samples//2)
    
    # Class 1: centered at (2, 2)
    X_class1 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
    y_class1 = np.ones(n_samples//2)
    
    # Combine
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_decision_boundary(X, y, model):
    """Visualize the decision boundary"""
    plt.figure(figsize=(15, 4))
    
    # Plot 1: Decision Boundary
    plt.subplot(1, 3, 1)
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh grid
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and data points
    plt.contourf(xx, yy, Z, alpha=0.4, levels=20, cmap='RdYlBu')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', edgecolors='k', alpha=0.7)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', edgecolors='k', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.colorbar(label='Probability')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost Function Convergence
    plt.subplot(1, 3, 2)
    plt.plot(model.cost_history, color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (Binary Cross-Entropy)')
    plt.title('Cost Function Convergence')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sigmoid Function Visualization
    plt.subplot(1, 3, 3)
    z = np.linspace(-10, 10, 100)
    sigmoid_values = model.sigmoid(z)
    plt.plot(z, sigmoid_values, color='purple', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.xlabel('z (wx + b)')
    plt.ylabel('σ(z)')
    plt.title('Sigmoid Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
# Execution:-

if __name__ == "__main__":
    print("=" * 60)
    print("LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=200)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")
    
    # Split data into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Create and train the model
    print("\n2. Training the model...")
    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X_train, y_train)
    
    # Display learned parameters
    print("\n3. Learned Parameters:")
    print(f"   Weights: {model.weights}")
    print(f"   Bias: {model.bias:.4f}")
    
    # Make predictions
    print("\n4. Making predictions on test set...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Calculate detailed metrics
    print("\n5. Detailed Metrics (Test Set):")
    metrics = calculate_metrics(y_test, y_pred_test)
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    
    print("\n   Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   TP={cm['TP']}, FP={cm['FP']}")
    print(f"   FN={cm['FN']}, TN={cm['TN']}")
    
    # Example predictions with probabilities
    print("\n6. Sample Predictions:")
    sample_indices = [0, 1, 2]
    for idx in sample_indices:
        prob = model.predict_proba(X_test[idx:idx+1])[0]
        pred = model.predict(X_test[idx:idx+1])[0]
        actual = y_test[idx]
        print(f"   Sample {idx}: Prob={prob:.4f}, Predicted={pred}, Actual={int(actual)}")
    
    # Visualize results
    print("\n7. Plotting results...")
    plot_decision_boundary(X, y, model)
    
print("Logistic Regression Implemented Sucessfully!")