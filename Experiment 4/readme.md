# Three-Layer Neural Network for MNIST Classification using TensorFlow (No Keras)

## Objective
Implement a three-layer neural network using TensorFlow (without Keras) to classify the MNIST handwritten digits dataset. This implementation demonstrates:

- Feed-forward propagation
- Backpropagation for training
- Model evaluation and performance analysis

## Description
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels. The task is to correctly classify these digits using a three-layer neural network built exclusively with TensorFlow operations.

## Network Architecture
- **Input Layer**: 784 neurons (flattened 28x28 pixel images)
- **Hidden Layer 1**: Configurable neurons (ReLU activation)
- **Hidden Layer 2**: Configurable neurons (ReLU activation)
- **Output Layer**: 10 neurons with softmax activation (for digit classes 0–9)

## Key Steps in the Implementation
### 1. Data Preparation
- Load the MNIST dataset using `tensorflow.keras.datasets` (only for data loading).
- Normalize the pixel values to the range [0, 1].
- Convert labels to one-hot encoded format.

### 2. Model Initialization
- Initialize weights and biases for each layer using TensorFlow's `GlorotUniform` initializer.
- Define placeholders for input images and labels.

### 3. Feed-Forward Propagation
Compute the output of each layer using:
- Matrix multiplication (`tf.matmul`)
- ReLU activation (`tf.nn.relu`)
- Softmax activation (`tf.nn.softmax`) for the output layer

### 4. Loss Function
Use cross-entropy loss (`tf.nn.softmax_cross_entropy_with_logits`) to measure prediction error.

### 5. Backpropagation
- Compute gradients using TensorFlow's `tf.GradientTape`.
- Update weights and biases using Stochastic Gradient Descent (SGD).

### 6. Evaluation
- Calculate accuracy on the test set.
- Print loss and accuracy at each training epoch.

## Hyperparameter Variations
Each model is trained with the following configurations:
- **Hidden Layers**: (160,100), (100,100), (100,160), (60,60), (100,60)
- **Learning Rates**: 0.01, 0.1, 1

## Output
### Sample Output after Training
```
Epoch 11/20, Loss: 0.2484, Test Accuracy: 0.9410
Epoch 12/20, Loss: 0.2421, Test Accuracy: 0.9439
Epoch 13/20, Loss: 0.2367, Test Accuracy: 0.9456
Epoch 14/20, Loss: 0.2320, Test Accuracy: 0.9469
Epoch 15/20, Loss: 0.2276, Test Accuracy: 0.9491
Epoch 16/20, Loss: 0.2237, Test Accuracy: 0.9504
Epoch 17/20, Loss: 0.2198, Test Accuracy: 0.9522
Epoch 18/20, Loss: 0.2161, Test Accuracy: 0.9536
Epoch 19/20, Loss: 0.2129, Test Accuracy: 0.9550
Epoch 20/20, Loss: 0.2104, Test Accuracy: 0.9564
Final Test Accuracy: 0.9564
```

## Performance Metrics
For each configuration, the following metrics are recorded:
- **Loss Curve**: Plots the training loss over epochs.
- **Accuracy Curve**: Plots test accuracy over epochs.
- **Final Test Accuracy**: Evaluates the model's effectiveness.
- **Confusion Matrix**: Visualizes classification performance.
- **Execution Time**: Measures the time taken to train the model.

## Results Visualization
- Loss and accuracy curves are plotted for each model variation.
- Confusion matrices visualize classification errors.
- Execution time is printed to assess computational efficiency.

## Dependencies
Ensure the following Python libraries are installed:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Running the Experiment
Execute the script in a Python environment:
```bash
python mnist_nn_experiments.py
```

## Conclusion
This experiment demonstrates how neural networks can be implemented using TensorFlow's core operations, without Keras. By analyzing results across different architectures and learning rates, we determine the optimal configurations for MNIST digit classification.

