# **Three-Layer Neural Network for MNIST Classification using TensorFlow (No Keras)**

---

## **Objective**

Implement a **three-layer neural network** using the **TensorFlow** library (**without Keras**) to classify the **MNIST handwritten digits** dataset. The implementation will demonstrate:
- **Feed-forward propagation**
- **Backpropagation for training**

---

## **Description**

The **MNIST dataset** consists of 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels. The task is to correctly classify these digits using a **three-layer neural network** built exclusively with **TensorFlow** operations.

### **Network Architecture**

- **Input Layer:** 784 neurons (28x28 pixel images flattened)
- **Hidden Layer 1:** 128 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation
- **Output Layer:** 10 neurons with softmax activation (for digit classes 0–9)

---

## **Key Steps in the Implementation**

### **1. Data Preparation**
- Load the **MNIST dataset** using `tensorflow.keras.datasets` (only for data loading).
- Normalize the pixel values to the range `[0, 1]`.

### **2. Model Initialization**
- Initialize **weights** and **biases** for each layer using TensorFlow's random functions.
- Define **placeholders** for input images and labels.

### **3. Feed-Forward Propagation**
- Compute the output of each layer using:
  - Matrix multiplication (`tf.matmul`)
  - ReLU activation (`tf.nn.relu`)
  - Softmax activation (`tf.nn.softmax`) for the output layer.

### **4. Loss Function**
- Use **cross-entropy loss** to measure prediction error.

### **5. Backpropagation**
- Compute gradients using **TensorFlow's `tf.GradientTape`**.
- Update weights and biases using **gradient descent optimization**.

### **6. Evaluation**
- Calculate **accuracy** on the test set.
- Print **loss** and **accuracy** at each training epoch.

---

## **Output**

### **Sample Output after Training:**
```
Epoch 1, Loss: 0.3805, Test Accuracy: 0.9103
Epoch 2, Loss: 0.2458, Test Accuracy: 0.9265
Epoch 3, Loss: 0.1974, Test Accuracy: 0.9361
Epoch 4, Loss: 0.1637, Test Accuracy: 0.9435
Epoch 5, Loss: 0.1402, Test Accuracy: 0.9490

Training completed.
Final Test Accuracy: 0.9490

```

---

## **Performance**

-  **Final Test Accuracy:** ~95%
-  **Efficient training** using manual implementation of feed-forward and backpropagation.
-  **Dynamic graph execution** leveraging TensorFlow's low-level APIs.

---

## **My Comments**

-  The implementation highlights how **TensorFlow's core operations** can be used to build neural networks **without Keras**.

-  The **manual backpropagation** approach provides deeper insights into the underlying gradient computations.


