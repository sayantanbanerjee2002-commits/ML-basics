import numpy as np
import matplotlib.pyplot as plt

class SimpleSVM:
    """
    Simple Support Vector Machine implementation from scratch
    Usinng  Gradient Descent to optimize the decision boundary
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        """
        Initialize SVM parameters
        
        learning_rate: How fast line is updated 
        lambda_param: Regularization of weight
        n_iterations: How many times line will update
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None  # Weights:- direction of line
        self.b = None  # Bias:- position of line
    
    def fit(self, X, y):
        """
        Train the SVM model
        
        X: Training data(ndarray of shape(n_features))
        y: Target_labels(between -1 and 1)
        """
        n_samples, n_features = X.shape
        
        # Converts the mismatched labels in -1 and 1
        y_ = np.where(y <= 0, -1, 1) # condition checking in ndarray
        self.w = np.ones(n_features)
        self.b = 0.05
        
        # Training loop using Gradient Descent:-
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):# here idx means index of dataset X
                # Check if point is correctly Classified with marginal line:
                # condition: y_i * (w·x_i + b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # if condition is true, point is perfectly placed ,no small updates or major updates require
                    # Only update weights for regularization
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Point is misclassified or too close
                    # Update both weights and bias
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} completed")
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Returns: array of Predicted labels(-1 or 1)
        """
        # Calculate w·x + b for data
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
    def visualize(self, X, y):
        """
        Visualize the decision boundary and data points (only for 2D data)
        """
        if X.shape[1] != 2:
            print("Visualization only works for 2D data!")
            return
        
        # Create a mesh to plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Predict for each point in mesh
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        
        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                   edgecolors='k', s=100, linewidth=1.5)
        
        # Draw decision boundary
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Create grid for hyperplane
        xx_line = np.linspace(xlim[0], xlim[1], 100)
        yy_line = -(self.w[0] * xx_line + self.b) / self.w[1]
        
        plt.plot(xx_line, yy_line, 'k-', linewidth=2, label='Decision Boundary')
        
        # Draw margins
        margin = 1 / np.sqrt(np.sum(self.w ** 2))
        yy_up = yy_line + margin
        yy_down = yy_line - margin
        plt.plot(xx_line, yy_up, 'k--', linewidth=1, label='Margin')
        plt.plot(xx_line, yy_down, 'k--', linewidth=1)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('SVM Decision Boundary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Create sample data
np.random.seed(42)

# Class 1: Points around (2, 2)
X1 = np.random.randn(20, 2) + np.array([2, 2])
y1 = np.ones(20)

# Class 2: Points around (-2, -2)
X2 = np.random.randn(20, 2) + np.array([-2, -2])
y2 = -np.ones(20)

# Combine data
X = np.vstack([X1, X2])
y = np.hstack([y1, y2])

# Create and train SVM
print("Training SVM...")
svm = SimpleSVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)

# Calculate accuracy
accuracy = np.mean(predictions == np.where(y <= 0, -1, 1))
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# Visualize results
print("\nVisualizing decision boundary...")
svm.visualize(X, y)

# Test on new data
print("\n--- Testing on New Data ---")
new_data = np.array([[3, 3], [-3, -3], [0, 0]])
new_predictions = svm.predict(new_data)
print(f"New data points: {new_data}")
print(f"Predictions: {new_predictions}")
print("(1 = Class 1, -1 = Class 2)")