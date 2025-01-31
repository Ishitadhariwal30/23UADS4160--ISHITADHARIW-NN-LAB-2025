import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        # Initialize weights (including bias)
        self.weights = np.zeros(input_size + 1)  # Including bias weight
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        # Sigmoid activation function (logistic regression)
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        # Add bias term to input
        x = np.insert(x, 0, 1)  # Inserting 1 at the beginning (bias)
        # Compute output using the sigmoid function
        return self.activation_function(np.dot(self.weights, x))

    def compute_error(self, X, y):
        # Compute the mean squared error (MSE) for Gradient Descent
        predictions = np.array([self.predict(x) for x in X])
        return np.mean((predictions - y) ** 2)

    def train(self, X, y):
        # Training process using Gradient Descent
        X = np.array(X)
        y = np.array(y)
        
        # Record error values for visualization
        error_history = []
        
        for epoch in range(self.epochs):
            gradient = np.zeros_like(self.weights)  # Initialize gradient
            for i in range(len(X)):
                x = np.insert(X[i], 0, 1)  # Add bias term
                y_pred = self.predict(X[i])  # Predicted output
                # Compute the gradient for each sample
                error = y_pred - y[i]
                gradient += error * x  # Accumulate gradient
            
            # Update weights based on the gradient
            self.weights -= self.learning_rate * gradient / len(X)
            
            # Compute and record the error for this epoch
            error_history.append(self.compute_error(X, y))
            
            # Optional: print the error for every 10th epoch
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Error: {error_history[-1]:.4f}")
        
        return error_history

# Example usage
if __name__ == "__main__":
    # Dataset for AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    y = np.array([0, 0, 0, 1])  # Expected outputs (AND gate)

    # Create Perceptron object
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)

    # Train the perceptron
    error_history = perceptron.train(X, y)

    # Print the final weights
    print("Final trained weights:", perceptron.weights)

    # Test the perceptron on the dataset
    for input_data in X:
        print(f"Input: {input_data}, Predicted Output: {perceptron.predict(input_data):.4f}")
    
    # Visualize the error curve using pandas and matplotlib (optional)
    error_df = pd.DataFrame({'Epoch': range(1, len(error_history) + 1), 'Error': error_history})
    error_df.plot(x='Epoch', y='Error', kind='line', title="Error over Epochs", xlabel="Epoch", ylabel="Error")
