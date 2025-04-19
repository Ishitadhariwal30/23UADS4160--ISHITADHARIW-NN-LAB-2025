

#  Convolutional Neural Network (CNN) for Fashion MNIST Classification

##  Objective
The goal of this project is to train and evaluate a Convolutional Neural Network (CNN) using the Keras library to classify images from the Fashion MNIST dataset. The project also explores the effect of various hyperparameters such as filter size, regularization strength, batch size, and optimization algorithms on model performance.

---

##  Dataset
We use the **Fashion MNIST** dataset, which includes:

- **70,000 grayscale images** (28x28 pixels)
- **10 categories** of clothing:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

**Split:**
- Training set: 60,000 images  
- Test set: 10,000 images

---

##  Model Architecture

The CNN model follows this structure:

1. **Conv2D Layer**: 32 filters, kernel size of (3,3) or (5,5), ReLU activation
2. **MaxPooling2D**: Pooling window of (2,2)
3. **Conv2D Layer**: 64 filters, ReLU activation
4. **MaxPooling2D**
5. **Flatten Layer**
6. **Dense Layer**: 128 units, ReLU activation
7. **Dropout**: Rate of 0.5
8. **Dense Output Layer**: 10 units, softmax activation

 **L2 Regularization** is used to reduce overfitting.  
 **Dropout** adds further regularization.

---

##  Code Overview

###  Libraries Used
- **TensorFlow / Keras** – Model building and training
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Visualization
- **scikit-learn** – Evaluation (confusion matrix)
- **Google Colab’s files module** – For downloading output plots

###  Code Flow
- **GPU Check**: Detects and utilizes available GPU
- **Data Loading & Preprocessing**: Normalizes image pixel values and reshapes input for CNN
- **Model Creation**: Based on varying hyperparameters
- **Training & Evaluation**: Models are trained and validated for different settings
- **Result Visualization**: Accuracy/loss plots and confusion matrix generation

---

## ⚙️ Hyperparameter Configurations Explored
- **Filter Sizes**: (3,3) and (5,5)
- **Regularizations**: 0.0001 and 0.001
- **Batch Sizes**: 32 and 64
- **Optimizers**: Adam and SGD
- **Dropout Rate**: 0.5

Each configuration is trained for **5 epochs**.

---

##  Performance Evaluation

###  Key Observations:

- **Filter Size**:  
  - Larger filters (5x5) capture more complex features.  
  - May slow down training and risk overfitting.

- **Regularization (L2)**:  
  - Helps reduce overfitting.  
  - Lower regularization value (0.0001) performed better than 0.001.

- **Batch Size**:  
  - Batch size of **32** provided better generalization and higher validation accuracy than 64.

- **Optimizer**:  
  - **Adam** outperformed SGD due to its adaptive learning rate.  
  - **SGD** converged slower but showed stable performance.

---

##  Best Performing Model

| Hyperparameter     | Value      |
|--------------------|------------|
| Filter Size        | 5          |
| Regularization     | 0.0001     |
| Batch Size         | 32         |
| Optimizer          | Adam       |

 This setup achieved the **highest validation accuracy (~90%)**.

---

##  Visual Results

- Training vs. Validation Accuracy and Loss curves
- Confusion Matrix for the best-performing model
- Line plot comparing validation accuracy across configurations

> All plots are saved and available for download.

---

##  Project Structure
```
├── model_training.py / .ipynb   # Code implementation
├── plots/
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   └── confusion_matrix.png
├── README.md                    # This file
```

---

##  Comments
- Exploring hyperparameters significantly enhances model understanding.
- Adam optimizer combined with smaller regularization and batch size gives a strong baseline.
- Visual analysis is crucial for diagnosing model behavior and guiding further tuning.

