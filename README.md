# Fashion-MNIST MLP Classifier

## Description
This repository contains an **optimized Multi-Layer Perceptron (MLP) model** for classifying images from the **Fashion-MNIST dataset** using **TensorFlow/Keras**.  

The main goal was to explore MLPs on image data, implement best practices to improve performance, and understand concepts like regularization, normalization, and learning rate scheduling.

> Note: While **Convolutional Neural Networks (CNNs)** generally perform better on image data, here I used an **MLP for experimentation and learning purposes**.

---

## Dataset
**Fashion-MNIST** is a dataset of **28x28 grayscale images of 10 clothing categories**:

- Total training images: 60,000  
- Validation split: 5,000  
- Test images: 10,000  
- Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

Data was normalized to values between 0 and 1.

---

## Model Architecture

The MLP model used the following architecture:

Flatten Layer
Dense(512, ReLU) + He Initialization + L2 Regularization
Batch Normalization
Dropout(0.2)

Dense(256, ReLU) + He Initialization + L2 Regularization
Batch Normalization
Dropout(0.2)

Dense(50, ReLU) + He Initialization + L2 Regularization
Batch Normalization

Dense(10, Softmax)


**Key Techniques Used:**
- **Batch Normalization** → stabilizes and accelerates training  
- **Dropout** → reduces overfitting  
- **L2 Regularization** → reduces overfitting  
- **He Initialization** → better convergence with ReLU  
- **Learning Rate Scheduler (ReduceLROnPlateau)** → fine-tunes learning when validation loss plateaus  
- **EarlyStopping** → stops training when validation loss stops improving  

---

## Training Progress & Improvements

1. **Initial MLP (baseline)**  
   - Simple 3-layer MLP  
   - Training accuracy: ~91%  
   - Test accuracy: ~81%  
   - Observation: Overfitting evident (high train acc, low test acc)

2. **Improvements Added:**  
   - I increased neurons (300→512, 80→256, 50→128)  
   - I added **Batch Normalization** and **Dropout**  
   - I applied **L2 regularization** and **He initialization**  
   - I used **ReduceLROnPlateau** callback and EarlyStopping  

3. **Result after improvements:**  
   - Training accuracy remained high (~89%)  
   - Test accuracy improved slightly (~85%)  
   - Overfitting reduced, model generalizes better  

> Note: Even after improvements, **MLP is limited for image data**. CNNs are generally recommended for achieving **>92% accuracy on Fashion-MNIST**.  


