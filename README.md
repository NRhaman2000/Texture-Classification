# Texture Classification: Classical vs. Deep Learning Approaches

This repository contains a computer vision pipeline that compares classical hand-crafted feature extraction against modern deep learning techniques for texture classification. 

The project utilizes the **Describable Textures Dataset (DTD)**, which contains 47 distinct texture categories (e.g., woven, banded, bubbly, cracked).

## Project Overview

This project serves as an exploration of how different machine learning architectures handle complex, highly variable texture data. The experiment is broken down into four distinct stages:

1. **The Classical Baseline:** Uses a Gabor Filter Bank to extract hand-crafted mathematical summaries of textures, classified by a Support Vector Machine (SVM).
2. **The Shallow CNN:** A basic Convolutional Neural Network trained from scratch to learn its own texture filters.
3. **The Over-Regularized CNN:** An attempt to fix overfitting using heavy Data Augmentation and 50% Dropout (which resulted in underfitting).
4. **The Balanced CNN:** The final, optimized architecture utilizing 3 convolutional layers, gentle augmentations, and 20% Dropout.

## Requirements

To run this project locally, you will need Python 3.x installed along with the following libraries. 

Run this command in your terminal to install the dependencies:

> pip install torch torchvision scikit-learn scikit-image matplotlib numpy scipy

*Note: for Mac/Linux it is "pip3" and for Windows it is "pip".*
*Note: The PyTorch code is optimized to automatically detect and utilize Apple Silicon (`mps`), Nvidia GPUs (`cuda`), or default to `cpu` if no hardware accelerator is found.*

## How to Run the Project

You must run the scripts in the following order. The first script downloads the dataset, and the subsequent scripts evaluate the different models.

### 1. Download and Prepare the Data
Run the main script first. This will automatically download the DTD dataset via `torchvision` and verify that the data loader is working by displaying a sample batch of 4 images.
`python Tex_class.py`

### 2. Run the Classical Pipeline
This script rubs a mathematical Gabor filter bank over the dataset to extract features and trains an SVM. It will output the accuracy and display a confusion matrix.
`python Tex_class_classical.py`

### 3. Run the Deep Learning Baseline
This script trains a simple 2-layer CNN. You will notice high training accuracy but poor test accuracy due to extreme overfitting.
`python Tex_class_cnn.py`

### 4. Run the Improved (Over-Regularized) CNN
This script applies heavy data augmentation (rotations, flips) and 50% dropout. The model will underfit and default to random guessing.
`python Tex_class_cnn_improved.py`

### 5. Run the Final Balanced CNN
This is the final, optimized model. It uses a 3-layer architecture, gentle horizontal flipping, and 20% dropout to achieve the best generalization on the test set.
`python Tex_class_cnn_balanced.py`

## Results Summary

| Model | Architecture Details | Final Accuracy |
| :--- | :--- | :--- |
| **Classical Pipeline** | Gabor Filters + SVM | ~12.18% |
| **Baseline CNN** | 2-Layer CNN (No Regularization) | ~11.33% |
| **Improved CNN** | 2-Layer CNN + Heavy Augmentation + 50% Dropout | ~7.61% |
| **Balanced CNN** | 3-Layer CNN + Gentle Augmentation + 20% Dropout | ~18.99% |

The results successfully demonstrate that a properly balanced Convolutional Neural Network, capable of learning its own feature filters, outperforms classical hand-crafted mathematical filters for complex texture classification tasks.