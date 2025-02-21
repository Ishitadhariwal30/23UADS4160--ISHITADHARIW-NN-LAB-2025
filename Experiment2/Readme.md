 Objective
Implement a multi-layer perceptron (MLP) network with one hidden layer using NumPy in Python to learn the XOR Boolean function. The implementation uses a step activation function and does not include backpropagation.
 
Description
This code demonstrates a manually configured MLP capable of solving the XOR problem. The XOR function is not linearly separable and requires at least one hidden layer in a neural network. The MLP has:
· Two input neurons
· One hidden layer with two neurons
· One output neuron
 
The network uses a step activation function and manually chosen weights and biases, demonstrating that even without backpropagation, an MLP can solve XOR with appropriate configuration.
 
 Key Points
· Uses NumPy for matrix operations.
· Step activation function (binary output).
· XOR dataset with inputs: [0,0], [0,1], [1,0], and [1,1].
· Manually selected weights and biases for correct XOR classification.
· No backpropagation; forward pass only.
 
Output
XOR Function Results:
Input: [0 0], Predicted Output: 0, Expected Output: 0
Input: [0 1], Predicted Output: 1, Expected Output: 1
Input: [1 0], Predicted Output: 1, Expected Output: 1
Input: [1 1], Predicted Output: 0, Expected Output: 0
 
Accuracy on XOR dataset: 100.0%
 
 
PERFORMACE
· Achieves 100% accuracy on the XOR dataset
· The model is computationally efficient due to simple matrix operations and a small network size.
· No training time since weights are manually assigned.
 
 My Comments
· The manual selection of weights and biases highlights how specific configurations can solve non-linear problems like XOR without backpropagation.
· This approach is useful for understanding MLP architecture but lacks scalability for larger datasets or more complex functions.
· Future improvements could include adding backpropagation for learning weights dynamically and exploring different activation functions for broader applications.
 Objective
Implement a multi-layer perceptron (MLP) network with one hidden layer using NumPy in Python to learn the XOR Boolean function. The implementation uses a step activation function and does not include backpropagation.
 
Description
This code demonstrates a manually configured MLP capable of solving the XOR problem. The XOR function is not linearly separable and requires at least one hidden layer in a neural network. The MLP has:
· Two input neurons
· One hidden layer with two neurons
· One output neuron
 
The network uses a step activation function and manually chosen weights and biases, demonstrating that even without backpropagation, an MLP can solve XOR with appropriate configuration.
 
 Key Points
· Uses NumPy for matrix operations.
· Step activation function (binary output).
· XOR dataset with inputs: [0,0], [0,1], [1,0], and [1,1].
· Manually selected weights and biases for correct XOR classification.
· No backpropagation; forward pass only.
 
Output
XOR Function Results:
Input: [0 0], Predicted Output: 0, Expected Output: 0
Input: [0 1], Predicted Output: 1, Expected Output: 1
Input: [1 0], Predicted Output: 1, Expected Output: 1
Input: [1 1], Predicted Output: 0, Expected Output: 0
 
Accuracy on XOR dataset: 100.0%
 
 
PERFORMACE
· Achieves 100% accuracy on the XOR dataset
· The model is computationally efficient due to simple matrix operations and a small network size.
· No training time since weights are manually assigned.
 
 My Comments
· The manual selection of weights and biases highlights how specific configurations can solve non-linear problems like XOR without backpropagation.
· This approach is useful for understanding MLP architecture but lacks scalability for larger datasets or more complex functions.
· Future improvements could include adding backpropagation for learning weights dynamically and exploring different activation functions for broader applications.
