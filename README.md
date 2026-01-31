# MNIST Neural Network (NumPy) — Load PyTorch Weights + Visualize

This project contains a fully NumPy-based feedforward neural network for MNIST classification.
Instead of training inside NumPy, the network **loads a trained PyTorch model (`mnist_model.pth`)** and copies the weights into NumPy matrices.  
You can then:
- run predictions on MNIST test images
- visualize the weight matrices as heatmaps

## Features
- Forward pass: **ReLU** (hidden layers) + **Softmax** (output layer)
- Loss: **Cross Entropy**
- Optional Backprop skeleton (included in code)
- Load weights from **PyTorch state_dict**
- Visualize predictions + weight matrices

---

## Requirements

### Python version
- Python **3.9+** recommended

### Dependencies
- numpy
- matplotlib
- seaborn
- torch
- torchvision


# Herleitung
# Feedforward Neural Network – Mathematical Overview (MNIST)

This project implements a simple fully-connected neural network for classifying handwritten digits from the **MNIST dataset**.

---

## 1. Input and Output Dimensions

Each MNIST image has size:

`28 × 28 = 784`

Thus the input is flattened into a vector:

$x \in \mathbb{R}^{784}$

The network predicts one of **10 classes** (digits 0–9):

$y \in \{0,\dots,9\}$

---

## 2. Network Architecture

The network consists of:

- Input Layer: 784
- Hidden Layer: 512
- Output Layer: 10

---

### Weight Matrices

$W_1 \in \mathbb{R}^{784 \times 512}$

$W_2 \in \mathbb{R}^{512 \times 10}$

---

### Bias Vectors

$b_1 \in \mathbb{R}^{512}$

$b_2 \in \mathbb{R}^{10}$

Bias terms act as learnable offsets and shift activation thresholds.

---

## 3. Forward Pass

### Hidden Layer Computation

Linear transformation:

$z_1 = xW_1 + b_1$

Activation with ReLU:

$a_1 = \mathrm{ReLU}(z_1)$

---

### ReLU Activation Function

$\mathrm{ReLU}(x) = \max(0,x)$

Negative values are removed:

$[-1,2,-3,4] \mapsto [0,2,0,4]$

ReLU introduces non-linearity, allowing the network to learn complex decision boundaries.

---

### Output Layer Computation

$z_2 = a_1W_2 + b_2$

The vector $z_2$ contains the **logits** (raw scores).

---

## 4. Softmax Output Probabilities

The Softmax function transforms logits into a probability distribution:

$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$

Properties:

- $\hat{y}_i > 0$
- $\sum_{i=1}^{10} \hat{y}_i = 1$

Thus:

$\hat{y} \in [0,1]^{10}$

---

## 5. Loss Function – Cross Entropy

For a one-hot encoded label $y$ and prediction $\hat{y}$:

$L = -\sum_{i=1}^{10} y_i \log(\hat{y}_i)$

Since $y$ is one-hot, this reduces to:

$L = -\log(\hat{y}_{\text{true}})$

Example:

- Prediction: $\hat{y} = [0.1,0.7,0.2]$
- True label: $y = [0,1,0]$

Loss:

$L = -\log(0.7)$

---

## 6. Complete Pipeline

$x \rightarrow z_1 \rightarrow a_1 \rightarrow z_2 \rightarrow \hat{y} \rightarrow L$

---

### Full Equation System

$z_1 = xW_1 + b_1$

$a_1 = \mathrm{ReLU}(z_1)$

$z_2 = a_1W_2 + b_2$

$\hat{y} = \mathrm{Softmax}(z_2)$

$L = -\sum_{i=1}^{10} y_i\log(\hat{y}_i)$

---

This defines the full forward computation for MNIST digit classification.



![Image](https://github.com/user-attachments/assets/9a4c7b4b-63ff-448c-9d11-3a69f83d2d7b)

![Image (1)](https://github.com/user-attachments/assets/4c082e2f-d6de-4af7-bcf6-d071fce51579)

![Image (2)](https://github.com/user-attachments/assets/768a4d3a-0a6e-420c-bb6a-d73dc8708e42)

![Image (3)](https://github.com/user-attachments/assets/fc3e7202-943e-4a23-bfcf-9b0953c920a0)

![Image (4)](https://github.com/user-attachments/assets/0279f3d2-2c6a-4e66-b2a6-60a848d65a62)
