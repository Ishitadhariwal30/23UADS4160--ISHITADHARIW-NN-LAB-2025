
## **Objective**

This document provides the implementation of the **Perceptron Learning Algorithm** using **NumPy** in Python. The focus is on evaluating the performance of a **single perceptron** for two logical functions: **NAND** and **XOR** using their respective truth tables.

---

## **Description**

A **perceptron** is the simplest form of an artificial neural network consisting of:
- **Input nodes**
- **Weights**
- **Bias term**
- **Activation function**

### **Working Mechanism:**
- The **weighted sum** is calculated by multiplying inputs with weights and adding the bias.
- The **activation function** classifies the output:
  - If **weighted sum ≥ 0**, then **output = 1**
  - If **weighted sum < 0**, then **output = 0**

---

## **Description of Code**

### **1. Class Perceptron**
- **Initialization:** Weights and bias are initialized to zero.
- **Activation Function:** Classifies output based on the weighted sum.
- **Predict Function:** Uses the activation function to predict output.
- **Train Function:** Adjusts weights using the perceptron learning rule.
- **Evaluate Function:** Calculates the accuracy of the model.

### **2. Training and Evaluation for NAND Gate**
- The **NAND truth table** (`nand_X`) and corresponding **labels** (`nand_y`) are defined.
- The **perceptron** is trained using the `train()` method.
- The **evaluate()** function computes the model's accuracy.
- **Predictions** for all input combinations in the NAND truth table are displayed.

### **3. Training and Evaluation for XOR Gate**
- The **XOR truth table** (`xor_X`) and **labels** (`xor_y`) are defined.
- The **perceptron** is trained and evaluated on the XOR dataset.
- **Accuracy** is calculated, and predictions for all XOR inputs are printed.

---

## **Output**

### **NAND Gate Results:**
```
Training Perceptron for NAND Gate
NAND Perceptron Accuracy: 1.0
Predictions for NAND Truth Table:
Input: [0 0], Prediction: 1
Input: [0 1], Prediction: 1
Input: [1 0], Prediction: 1
Input: [1 1], Prediction: 0
```

### **XOR Gate Results:**
```
Training Perceptron for XOR Gate
XOR Perceptron Accuracy: 0.5
Predictions for XOR Truth Table:
Input: [0 0], Prediction: 1
Input: [0 1], Prediction: 1
Input: [1 0], Prediction: 0
Input: [1 1], Prediction: 0
```

---

## **Performance**

### **NAND Gate:**
-  **Accuracy:** 1.0 (100%)
-  **Predictions:** Correct for all input combinations.

### **XOR Gate:**
- **Accuracy:** Approximately 0.5 (50%)
- **Predictions:** Incorrect for half of the input combinations.

---

## **My Comments**

-  The **perceptron** performs **perfectly** on the **NAND gate**, demonstrating that it can handle **linearly separable** problems.

-  The **XOR gate** performance highlights a **key limitation**: the perceptron **cannot solve non-linearly separable** problems.

