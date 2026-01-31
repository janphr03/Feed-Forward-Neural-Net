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

Install locally with:

```bash
pip install numpy matplotlib seaborn torch torchvision





# Herleitung

Im Rahmen des Projekts wurden die zentralen mathematischen Grundlagen eines Feedforward-Neural-Networks ausführlich hergeleitet und dokumentiert. Dazu gehören:

- die Initialisierung
- der Forward Pass mit Gewichtsmatrizen, Bias-Termen und ReLU-Aktivierung
- die Softmax-Funktion zur Umwandlung der Modell-Outputs in Wahrscheinlichkeitsverteilungen
- der Cross-Entropy-Loss und seine Rolle bei Klassifikationsaufgaben

**=> Was noch fehlt** ist das Fundament der Backpropagation

Die Herleitungen dienen dazu, die Funktionsweise der einzelnen Komponenten transparent und mathematisch nachvollziehbar darzustellen. 
Dadurch basiert das Projekt auf einem klar verständlichen theoretischen Fundament, auch wenn die Backpropagation später über eine Bibliothek implementiert wird.

![Image](https://github.com/user-attachments/assets/9a4c7b4b-63ff-448c-9d11-3a69f83d2d7b)

![Image (1)](https://github.com/user-attachments/assets/4c082e2f-d6de-4af7-bcf6-d071fce51579)

![Image (2)](https://github.com/user-attachments/assets/768a4d3a-0a6e-420c-bb6a-d73dc8708e42)

![Image (3)](https://github.com/user-attachments/assets/fc3e7202-943e-4a23-bfcf-9b0953c920a0)

![Image (4)](https://github.com/user-attachments/assets/0279f3d2-2c6a-4e66-b2a6-60a848d65a62)
