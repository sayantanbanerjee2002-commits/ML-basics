import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression implemented from scratch using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the Linear Regression model
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []  # Track cost over iterations
    
    def fit(self, X, y):
        """
        Train the model using gradient descent:
        
        Parameters:
        
        X :ndarray of shape (n_samples, n_features) -> Training data
            
        y : ndarray of shape (n_samples,) -> Actual output
           
        """
        # Get number of samples and features
        n_samples, n_features = X.shape
        
        # Initialize weights and bias to zero
        self.weights = np.ones(n_features)
        self.bias = 1
        
        # Gradient Descent Algorithm Implementation:-
        for i in range(self.iterations):
            # Make predictions with current weights and bias
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            # dw = (1/n) * Σ 2x(predicted - actual)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            
            # db = (1/n) * Σ 2(predicted - actual)
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate and store cost for this iteration
            cost = self._calculate_cost(y, y_predicted)
            self.cost_history.append(cost)
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.4f}")
    
    def predict(self, X):
        """
        Make predictions using the learned weights
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
           data points for prediction
            
        Returns:
        predictions : ndarray of shape (n_samples,)
            Predicted values
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    
    def _calculate_cost(self, y_true, y_pred):
        """
        Calculate Mean Squared Error cost
        
        Parameters:-
        y_true : ndarray Actual Value 
        y_pred :ndarray predicted output 
        Returns:-
        cost : Mean Squared Error(Float)
        """
        n_samples = len(y_true)
        cost = (1/n_samples) * np.sum((y_pred - y_true) ** 2)
        return cost
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)
        
        Parameters:
        X : Input Featuers (ndarray)
        y : True Values(ndarray)  
        Returns:-
        r2_score : float value :- 1- SS_residual/SS_total
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2_score = 1 - (ss_residual / ss_total)
        return r2_score

# Testing :-
def generate_sample_data(n_samples=100): # Generate Sample data for testing
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(n_samples)
    return X, y


def plot_results(X, y, model): # Visualization of results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost function over iterations
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history, color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost Function Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Execution:-
if __name__ == "__main__":
    print("=" * 50)
    print("LINEAR REGRESSION FROM SCRATCH")
    print("=" * 50)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=100)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Create and train the model
    print("\n2. Training the model...")
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    
    # Display learned parameters
    print("\n3. Learned Parameters:")
    print(f"   Weight (slope): {model.weights[0]:.4f}")
    print(f"   Bias (intercept): {model.bias:.4f}")
    
    # Make predictions
    print("\n4. Making predictions...")
    sample_input = np.array([[1.5]])
    prediction = model.predict(sample_input)
    print(f"   Input: {sample_input[0][0]}")
    print(f"   Prediction: {prediction[0]:.4f}")
    
    # Calculate R² score
    r2 = model.score(X, y)
    print(f"\n5. Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Final Cost: {model.cost_history[-1]:.4f}")
    
    # Visualize results
    print("\n6. Plotting results...")
    plot_results(X, y, model)
    print("Implement Gradient Descent Sucessfully!")