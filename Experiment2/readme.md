

## **Objective**

This document outlines the implementation of a **multi-layer perceptron (MLP)** network with one hidden layer using **NumPy** in Python to learn the **XOR Boolean function**. The model employs a **step activation function** and operates **without backpropagation**.

---

## **Description**

The implementation demonstrates a **manually configured MLP** capable of solving the **XOR problem**, which is not linearly separable and requires at least one hidden layer in a neural network.

### **MLP Architecture**

- **Input Layer:** 2 neurons
- **Hidden Layer:** 1 hidden layer with 2 neurons
- **Output Layer:** 1 neuron

### **Key Characteristics**

- **Activation Function:** Step function (binary output)
- **Weights & Biases:** Manually selected for correct XOR classification
- **Learning Approach:** Forward pass only (No backpropagation)

---

## **Key Points**

- Utilizes **NumPy** for efficient matrix operations.
- Employs a **step activation function** for binary outputs.
- Works on the **XOR dataset** with the following inputs:
  - `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`
- Achieves accurate classification of XOR through **manually selected weights and biases**.
- Conducts only a **forward pass**, with **no backpropagation** involved.

---

## **Output**

```
Input: [0 0], Predicted Output: 0, Expected Output: 0
Input: [0 1], Predicted Output: 1, Expected Output: 1
Input: [1 0], Predicted Output: 1, Expected Output: 1
Input: [1 1], Predicted Output: 0, Expected Output: 0
```

### **Accuracy:**

```
Accuracy on XOR dataset: 100.0%
```

---

## **Performance**

- **Accuracy:** Achieves **100% accuracy** on the XOR dataset.
- **Efficiency:**
  - Optimized for computational efficiency with simple matrix operations and a compact network structure.
- **Training Time:**
  - **Zero training time** due to manual assignment of weights.

---

## **My Comments**

- The **manual selection** of weights and biases underscores how specific configurations can solve **non-linear problems** like XOR **without backpropagation**.

- Although this approach is **ideal for understanding MLP architecture**, it has certain **limitations**:
  - **Scalability issues** for larger datasets.
  - **Inability** to handle more complex functions effectively.



