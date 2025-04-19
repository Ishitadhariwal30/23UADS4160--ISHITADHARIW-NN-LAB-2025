# 7. WAP to Retrain a Pretrained ImageNet Model to Classify a Medical Image Dataset

## Overview

This project retrains a **pretrained VGG16** model (from ImageNet) to perform **binary classification** on a medical image dataset for Alzheimer's Disease, distinguishing between **Mild Dementia** and **Very Mild Dementia**.

Using transfer learning, only the top layers of VGG16 are customized while the base model remains frozen. Data augmentation, class balancing, and performance evaluation are implemented using TensorFlow and supporting libraries.

---

## Dataset

- **Path**: `C:\Users\LENOVO\Desktop\Python\Datasets\Alzeimehers\Data`
- **Classes**:
  - `Mild_Dementia`
  - `Very_mild_Dementia`
- **Image Counts**:
  - Training: 14,982 images
  - Validation: 3,745 images
- **Image Size**: 224x224 pixels

---

## Model Architecture

- **Base Model**: VGG16 (pretrained on ImageNet, `include_top=False`)
- **Top Layers**:
  - Flatten
  - Dense (256, ReLU)
  - Dropout (0.5)
  - Dense (1, Sigmoid) for binary classification

---

## Preprocessing and Augmentation

- Pixel Normalization: `rescale=1./255`
- Augmentations:
  - Horizontal Flip
  - Zoom Range: 20%
- Dataset Split: 80% Training, 20% Validation

---

## Class Balancing

To handle class imbalance, class weights were computed using scikit-learn:

```python
{0: 1.87, 1: 0.68}
