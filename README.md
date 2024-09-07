# MLP and CNN Classification with CIFAR-10

This repository contains the implementation and comparison of different machine learning classifiers, including **Multilayer Perceptrons (MLPs)** and **Convolutional Neural Networks (CNNs)**, for solving a classification problem using the **CIFAR-10 dataset**.

## Project Overview

The goal of this project is to compare the performance of **MLPs** and **CNNs** in solving an image classification task using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 color images of 32x32 pixels, categorized into 10 classes: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

### Algorithms Implemented
- **Multilayer Perceptrons (MLPs)**:
  - Various architectures tested with different numbers of hidden layers and neurons.
  - Techniques like **Dropout** were used to reduce overfitting.
  - Optimizer: **Adam**
  - Loss Function: **Categorical Crossentropy**

- **Convolutional Neural Networks (CNNs)**:
  - Tested various architectures including multiple convolutional layers followed by pooling layers.
  - ReLU activation was used in hidden layers, and **MaxPooling** and **AveragePooling** layers were compared.
  - Final CNN architecture incorporated **Dropout** to combat overfitting.

## Dataset

### CIFAR-10
The CIFAR-10 dataset contains:
- **60,000** color images of size 32x32, split into **50,000 training** images and **10,000 test** images.
- **10 classes**: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

## Models and Results

### MLP (Multilayer Perceptron)

- Multiple architectures with varying numbers of hidden layers (from 1 to 5) were tested.
- Best result achieved with an MLP of 5 hidden layers, reaching:
  - **Train Accuracy**: 81.72%
  - **Test Accuracy**: 56.63%
- Overfitting was reduced by using Dropout layers, but the overall performance was still limited compared to CNNs.

### CNN (Convolutional Neural Network)

- Several architectures of CNNs were tested, with the final model consisting of:
  - Convolutional layers with increasing numbers of filters (up to 256).
  - MaxPooling layers for dimensionality reduction.
  - Dropout layers to reduce overfitting.
- Final CNN architecture achieved:
  - **Train Accuracy**: 86.57%
  - **Test Accuracy**: 82.39%
  
This performance far exceeded that of the MLPs and other traditional classifiers like k-Nearest Neighbors (kNN) and Nearest Centroid (NC).

## Key Findings

- CNNs outperform MLPs in image classification tasks, especially for datasets like CIFAR-10, where spatial features are important.
- Techniques such as **Dropout** and **Batch Normalization** significantly improve model generalization.
- **MLPs** can still perform reasonably well, but they are less efficient than CNNs for image data.

## Report

For a detailed explanation of the project's objectives, methods, and results, please refer to the [full project report](Report.pdf).

## Dependencies

This project was implemented using **Python** and **Google Colaboratory**. To replicate the results, ensure you have the following packages installed:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

You can install these packages via pip:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn```

## Running the Code

- **Run `Convolutional.ipynb`** for the Convolutional Neural Network (CNN) implementation.
- **Run `Dense.ipynb`** for the Multilayer Perceptron (MLP) implementation.
