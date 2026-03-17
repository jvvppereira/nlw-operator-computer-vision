# LeNet-5 for MNIST Classification

This repository contains a PyTorch implementation of the classic **LeNet-5** convolutional neural network architecture, applied to the MNIST dataset for handwritten digit recognition.

## Project Overview

The project is implemented in a Jupyter Notebook (`lenet5.ipynb`) and covers the entire machine learning workflow, from model definition to visualization of results.

### Key Modifications
While following the original LeNet-5 spirit, this version includes modern improvements:
- **Activation Function**: Replaced Tanh with **ReLU** for faster convergence.
- **Pooling Layer**: Replaced Average Pooling with **Max Pooling**.
- **Padding**: Added `padding=2` to the first layer to handle 28x28 input images as if they were 32x32 (standard LeNet-5 input).

## Repository Structure

- `lenet5.ipynb`: Main notebook containing the model, training, and visualizations.
- `pyproject.toml`: Project dependencies managed by `uv`.
- `lenet5_mnist.pth`: Saved weights of the trained model (generated after training).

## Features

### 1. Model Architecture
A classic CNN with:
- 3 Convolutional layers.
- 2 Max Pooling layers.
- 2 Fully Connected layers.

### 2. Dataset Setup
- Automatic download of the **MNIST** dataset.
- Data conversion to tensors.
- Visualization of sample digits.

### 3. Training & Evaluation
- **Optimizer**: Adam.
- **Loss Function**: Cross-Entropy.
- **Epochs**: 5.
- **Evaluation**: Accuracy calculation on the 10,000-image test set.

### 4. Advanced Visualizations
- **Initial Filters**: View the random weights of the first layer (C1).
- **Error Analysis**: Display images that the network failed to classify correctly.
- **Feature Maps**: Visualize how an image is processed through the first layer's filters.

## Getting Started

### Prerequisites
Make sure you have PyTorch and Matplotlib installed. If you are using `uv`, you can simply run:
```bash
uv sync
```

### Running the Project
Open the `lenet5.ipynb` notebook in VS Code or Jupyter and run the cells sequentially:
1. Define the `LeNet5` class.
2. Load the dataset.
3. Train the model.
4. Evaluate performance.
5. Explore the visualizations.

## Results
The model typically achieves over **98% accuracy** on the test set after just 5 epochs of training.
