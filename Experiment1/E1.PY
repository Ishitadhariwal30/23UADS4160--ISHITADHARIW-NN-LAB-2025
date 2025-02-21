import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias term
        return self.activation(np.dot(self.weights, x))
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Add bias term
                prediction = self.activation(np.dot(self.weights, xi))
                self.weights += self.learning_rate * (target - prediction) * xi
    
    def evaluate(self, X, y):
        correct = sum(self.predict(xi) == yi for xi, yi in zip(X, y))
        return correct / len(y)

# NAND Truth Table
X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nand = np.array([1, 1, 1, 0])

# XOR Truth Table
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train and evaluate perceptron on NAND
t_perceptron = Perceptron(input_size=2)
t_perceptron.train(X_nand, y_nand)
nand_accuracy = t_perceptron.evaluate(X_nand, y_nand)
print(f"NAND Perceptron Accuracy: {nand_accuracy:.4f}")

# Train and evaluate perceptron on XOR
t_perceptron = Perceptron(input_size=2)
t_perceptron.train(X_xor, y_xor)
xor_accuracy = t_perceptron.evaluate(X_xor, y_xor)
print(f"XOR Perceptron Accuracy: {xor_accuracy:.4f}")
