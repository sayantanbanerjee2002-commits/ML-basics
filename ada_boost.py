import numpy as np
import matplotlib.pyplot as plt

#  WEAK LEARNER (DECISION STUMP):-

class DecisionStump:
    """
    A Decision Stump is the Single level decision tree.
    It makes a decision based on just one feature and one threshold.
    
    """
    
    def __init__(self):
        self.polarity = 1  # Direction of comparision (1 or -1)
        self.feature_idx = None  # index of used feature
        self.threshold = None  # value based on which we take decision
        self.alpha = None  # performance of the stump
    
    def predict(self, X):
        """
        Make predictions based on the described rule
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]  # Get the relevant feature
        
        predictions = np.ones(n_samples)  # Start with all 1s initially
        
        # Apply the rule based on polarity
        if self.polarity == 1:
            # Rule: if feature < threshold, predict -1
            predictions[X_column < self.threshold] = -1
        else: # polarity = -1
            # Rule: if feature > threshold, predict -1
            predictions[X_column > self.threshold] = -1
        
        return predictions


# ADABOOST ALGORITHM:-

class AdaBoost:
    """
    AdaBoost: Combines multiple weak classifiers  into a strong classifier
    """
    
    def __init__(self, n_clf=7):
        """
        Initialize AdaBoost
        
        n_clf: Number of weak learners (stumps) to train
        """
        self.n_clf = n_clf
        self.clfs = []  # List to store all weak learners
    
    def fit(self, X, y):
        """
        Train AdaBoost on data
        
        X: Training features (n_samples, n_features)
        y: Training labels (should be -1 or 1)
        """
        n_samples, n_features = X.shape
        
        #  Initialize weights:- all points equally important means they get eual weight 
        # Each point gets weight = 1/n ; sum of all the weights must be 1
        w = np.full(n_samples, (1 / n_samples))
        
        self.clfs = []  
        
        # Train n_clf weak learners:-
        for clf_idx in range(self.n_clf):
            print(f"\n--- Training Weak Learner {clf_idx + 1}/{self.n_clf} ---")
            
            # Create a new decision stump
            clf = DecisionStump()
            
            # Find the best stump for current weights
            min_error = float('inf')  # Start with infinite error
            
            # Try all features
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)  
                
                # Try both directions of inequality:-
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        # Make predictions with this rule
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1
                        
                        # Calculate weighted error:- sum of all the weights where my prediction is wrong 
                        misclassified = w[y != predictions]
                        error = sum(misclassified)
                        
                        # Keep track of best stump
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i
            
            #  Calculate performance of the stumps
            # Formula of performance of the stumps: alpha = 0.5 * ln((1 - error) / (error + 1e-10))
            # The 1e-10 prevents underfitting
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            # Make predictions with this stump
            predictions = clf.predict(X)
            
            #  Update weights:-
            # Increase weight of misclassified points, decrese weight of corrected data points
            # Formula: w = w * exp(-alpha)
            w *= np.exp(-clf.alpha )
            # Normalize weights so that sum of all weights result 1
            w /= np.sum(w)
            
            self.clfs.append(clf) # Store this Classifier
            
            print(f"Error: {min_error:.4f}")
            print(f"performance of the Stump: {clf.alpha:.4f}")
            print(f"Feature: {clf.feature_idx}, Threshold: {clf.threshold:.4f}")
    
    def predict(self, X):
        """
        Make predictions using all weak learners
        
        Final prediction = Majority vote of all stumps
        """
        #  Agregrate all classifiers:-
        # Each classifier votes with weight =Performance of the stumps
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]# List Comprehension
        
        # Sum all weighted votes
        y_pred = np.sum(clf_preds, axis=0)
        
        # Final prediction based on sign
        y_pred = np.sign(y_pred)
        
        return y_pred
    
    def visualize_stumps(self, X, y):
        """
        Visualize how each stump makes decisions (only for 2D data)
        """
        if X.shape[1] != 2:
            print("Visualization only works for 2D data!")
            return
        
        n_stumps = len(self.clfs)
        fig, axes = plt.subplots(1, n_stumps + 1, figsize=(4 * (n_stumps + 1), 4))
        
        # Plot each stump
        for idx, clf in enumerate(self.clfs):
            ax = axes[idx]
            
            # Plot data points
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                      edgecolors='k', s=100, linewidth=1.5)
            
            # Draw decision boundary
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            
            if clf.feature_idx == 0:
                # Vertical line
                ax.axvline(x=clf.threshold, color='black', 
                          linestyle='--', linewidth=2, label='Decision Boundary')
            else:
                # Horizontal line
                ax.axhline(y=clf.threshold, color='black', 
                          linestyle='--', linewidth=2, label='Decision Boundary')
            
            ax.set_title(f'Stump {idx + 1}\nAlpha: {clf.alpha:.2f}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot final combined prediction
        ax = axes[-1]
        predictions = self.predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm',
                  edgecolors='k', s=100, linewidth=1.5)
        ax.set_title('Final AdaBoost\nPrediction')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# EXAMPLE USAGE:-

# Create sample data
np.random.seed(42)

# Class 1: 
X1 = np.random.randn(30, 2) + np.array([2, 2])
y1 = np.ones(30)

# Class 2: 
X2 = np.random.randn(30, 2) + np.array([-2, -2])
y2 = -np.ones(30)

# Add some noise (harder to classify)
X3 = np.random.randn(10, 2)
y3 = np.array([1 if i % 2 == 0 else -1 for i in range(10)])

# Combine all data
X = np.vstack([X1, X2, X3])
y = np.hstack([y1, y2, y3])

# Shuffle data
shuffle_idx = np.random.permutation(len(y))
X, y = X[shuffle_idx], y[shuffle_idx]

print("=" * 60)
print("ADABOOST TRAINING")
print("=" * 60)

# Create and train AdaBoost
clf = AdaBoost(n_clf=7)  
clf.fit(X, y)

# Make predictions
print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

y_pred = clf.predict(X)

# Calculate accuracy
accuracy = np.mean(y_pred == y)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# Show some predictions
print("\n--- Sample Predictions ---")
for i in range(5):
    print(f"Point {i+1}: True={int(y[i]):2d}, Predicted={int(y_pred[i]):2d}")

# Visualize
print("\nGenerating visualization...")
clf.visualize_stumps(X, y)

# Test on new data
print("\n" + "=" * 60)
print("TESTING ON NEW DATA")
print("=" * 60)

new_data = np.array([[3, 3], [-3, -3], [0, 0], [2, -2]])
new_predictions = clf.predict(new_data)

print("\nNew data points and predictions:")
for i, (point, pred) in enumerate(zip(new_data, new_predictions)):
    print(f"Point {i+1}: {point} → Class {int(pred)}")
print("\n(1 = Class 1, -1 = Class 2)")