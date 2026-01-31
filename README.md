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

$\hat y_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$

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


---

# Backpropagation and Gradient Descent

Training a neural network means adjusting the parameters

- weights $W$
- biases $b$

such that the loss function $L$ becomes minimal.

This is done using **Backpropagation** (gradient computation) and **Gradient Descent** (parameter update).

---

## 1. Training Loop (One Epoch)

For each training sample $(x,y)$ the network performs:

1. Forward Pass  
2. Loss computation  
3. Backpropagation  
4. Weight and bias update  

Repeating this over the full dataset corresponds to **one epoch**.

---

## 2. Parameter Update Rule

The weights are updated using gradient descent:

$W \leftarrow W - \eta \frac{\partial L}{\partial W}$

Biases are updated similarly:

$b \leftarrow b - \eta \frac{\partial L}{\partial b}$

Where:

- $\eta$ = learning rate  
- $\frac{\partial L}{\partial W}$ = gradient of the loss w.r.t. weights  

The gradient tells us how strongly the loss changes when parameters change.

---

## 3. Gradient Definition

A partial derivative is formally defined as:

$\frac{\partial L}{\partial W} = \lim_{\Delta W \to 0} \frac{\Delta L}{\Delta W}$

It measures the sensitivity of the loss with respect to a parameter.

---

## 4. Backpropagation Principle

Backpropagation propagates the error from the output layer backwards through the network.

Each layer computes gradients using the **chain rule**:

$\frac{\partial L}{\partial W}
= \frac{\partial L}{\partial a}
\cdot \frac{\partial a}{\partial z}
\cdot \frac{\partial z}{\partial W}$

This allows efficient computation of all gradients.

---

## 5. Output Layer Error Term

For Softmax + Cross Entropy loss, the error simplifies to:

$\delta_2 = \hat{y} - y$

Where:

- $\hat{y}$ = predicted probability vector  
- $y$ = true one-hot label  

---

### Gradients for Output Layer Parameters

Weight gradient:

$\frac{\partial L}{\partial W_2} = a_1^T \delta_2$

Bias gradient:

$\frac{\partial L}{\partial b_2} = \delta_2$

---

## 6. Hidden Layer Error Term

The error is propagated backwards:

$\delta_1 = (\delta_2 W_2^T) \odot \mathrm{ReLU}'(z_1)$

Where $\odot$ denotes elementwise multiplication.

---

### ReLU Derivative

$\mathrm{ReLU}'(x) =
\begin{cases}
1 & x > 0 \\
0 & x \leq 0
\end{cases}$

Inactive neurons ($z \leq 0$) do not contribute to gradient flow.

---

### Gradients for Hidden Layer Parameters

Weight gradient:

$\frac{\partial L}{\partial W_1} = x^T \delta_1$

Bias gradient:

$\frac{\partial L}{\partial b_1} = \delta_1$

---

## 7. Complete Backpropagation Pipeline

Forward:

$x \rightarrow z_1 \rightarrow a_1 \rightarrow z_2 \rightarrow \hat{y}$

Backward:

$\delta_2 \rightarrow \delta_1 \rightarrow \nabla W_2,\nabla W_1$

---

## 8. Optimization View (Loss Landscape)

Training can be interpreted as minimizing a loss function over a high-dimensional parameter space:

$\theta = \{W_1,W_2,b_1,b_2\}$

The goal is:

$\theta^\* = \arg\min_\theta L(\theta)$

Gradient descent iteratively moves parameters toward a local minimum:

$\theta \leftarrow \theta - \eta \nabla_\theta L$

---

This defines the full mathematical foundation of backpropagation and learning in a feedforward neural network.




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
