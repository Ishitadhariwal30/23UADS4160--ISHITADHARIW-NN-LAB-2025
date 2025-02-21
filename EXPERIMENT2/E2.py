

import numpy as np

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])  # Expected XOR outputs

# Manually chosen weights and biases for XOR
def mlp_xor(x):
    # Hidden layer (2 neurons)
    w_hidden = np.array([[1, 1],   # Neuron 1
                         [1, 1]])  # Neuron 2
    b_hidden = np.array([-0.5, -1.5])  # Biases for hidden layer

    # Output layer (1 neuron)
    w_output = np.array([1, -2])  # Weights for output neuron
    b_output = -0.5  # Bias for output neuron

    # Forward pass
    hidden_input = np.dot(x, w_hidden.T) + b_hidden
    hidden_output = step_function(hidden_input)

    final_input = np.dot(hidden_output, w_output) + b_output
    final_output = step_function(final_input)

    return final_output

# Test the MLP
print("XOR Function Results:")
for inp, target in zip(X, y):
    output = mlp_xor(inp)
    print(f"Input: {inp}, Predicted Output: {output}, Expected Output: {target}")

# Check overall accuracy
outputs = np.array([mlp_xor(xi) for xi in X])
accuracy = np.mean(outputs == y) * 100
print(f"\nAccuracy on XOR dataset: {accuracy}%")
