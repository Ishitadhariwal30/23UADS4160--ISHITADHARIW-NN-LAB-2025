import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        # Initialize weights (including bias)
        self.weights = np.zeros(input_size + 1)  # Including bias weight
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        # Step function: returns 1 if x >= 0, otherwise 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Add bias term to input
        x = np.insert(x, 0, 1)  # Inserting 1 at the beginning (bias)
        # Compute the output
        return self.activation_function(np.dot(self.weights, x))

    def train(self, X, y):
        # Training process
        for epoch in range(self.epochs):
            for i in range(len(X)):
                # Add bias term to each input
                x = np.insert(X[i], 0, 1)  # Inserting 1 at the beginning (bias)
                y_pred = self.activation_function(np.dot(self.weights, x))
                # Update the weights based on error
                self.weights += self.learning_rate * (y[i] - y_pred) * x

# Example usage
if __name__ == "__main__":
    # Sample dataset for AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    y = np.array([0, 0, 0, 1])  # Expected outputs (AND gate)

    # Create Perceptron object
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)

    # Train the perceptron
    perceptron.train(X, y)

    # Print trained weights
    print("Trained weights:", perceptron.weights)

    # Test the perceptron on the dataset
    for input_data in X:
        print(f"Input: {input_data}, Predicted Output: {perceptron.predict(input_data)}")
