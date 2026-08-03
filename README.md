# 🚀 Texture Classification: From Classical Machine Learning to Deep Learning Ensembles

Welcome to the Texture Classification project! This repository contains a complete, end-to-end computer vision pipeline that compares classical hand-crafted feature extraction against state-of-the-art deep learning neural networks.

## 🧠 What is this project?

Identifying textures (like whether a surface is "woven," "bubbly," or "cracked") is a notoriously difficult task for computers. Unlike objects (which have clear shapes and edges), textures are defined by highly repetitive, fine-grained, and often overlapping visual patterns. 

This project explores how machine learning has evolved to solve this problem. It uses the **Describable Textures Dataset (DTD)**, which contains 47 distinct and highly complex texture categories.

By running the scripts in this repository, you will trace the evolution of computer vision:
1. **Classical Era:** Using mathematical formulas (Gabor Filters) and a Support Vector Machine (SVM).
2. **Early Deep Learning:** Building Custom Convolutional Neural Networks (CNNs) from scratch.
3. **The Modern Era:** Fine-tuning massive, pre-trained architectures (EfficientNetV2, ConvNeXt) and combining their "brains" into a Dual Ensemble for maximum accuracy.

---

## 📂 Project Structure

Here is a quick guide to what each script does. They are meant to be run in this order to see the accuracy improve!

| File Name | Description |
| :--- | :--- |
| `Tex_class.py` | **Data Setup:** Downloads the DTD dataset and tests the data loader. |
| `Tex_class_classical.py` | **Classical Baseline:** Extracts Gabor filters and trains an SVM. |
| `Tex_class_cnn.py` | **Basic CNN:** A simple 2-layer network (demonstrates overfitting). |
| `Tex_class_cnn_improved.py`| **Over-Regularized CNN:** Adds heavy augmentation and 50% dropout. |
| `Tex_class_cnn_balanced.py`| **Balanced CNN:** A fully optimized 3-layer custom network. |
| `Tex_class_transfer.py` | **ResNet-50:** Uses Transfer Learning to jumpstart accuracy. |
| `Tex_class_sota.py` | **EfficientNetV2:** True fine-tuning of a modern architecture. |
| `Tex_class_ultimate.py` | **ConvNeXt-Tiny:** Introduces Test-Time Augmentation (TTA). |
| `Tex_class_ensemble.py` | **The Final Boss:** Trains two modern models and averages their predictions. |

---

## ⚙️ Prerequisites & Installation

To run this project on your local machine, you will need **Python 3.8+** installed. 

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/NRhaman2000/Texture-Classification.git](https://github.com/NRhaman2000/Texture-Classification.git)
cd Texture-Classification