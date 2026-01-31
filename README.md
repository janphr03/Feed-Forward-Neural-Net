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


# Feedforward Neural Network – Mathematische Herleitung (MNIST)

Dieses Projekt implementiert ein einfaches vollständig verbundenes neuronales Netz
zur Klassifikation handgeschriebener Ziffern aus dem **MNIST-Datensatz**.

---

## 1. Input- und Output-Dimensionen

Ein MNIST-Bild hat die Größe:

`28 × 28 = 784`

Deshalb wird das Bild zu einem Vektor umgeformt:

$x \in \mathbb{R}^{784}$

Das Netz soll eine von **10 Klassen** erkennen (Ziffern 0–9):

$y \in \{0,\dots,9\}$

---

## 2. Netzwerk-Architektur

Das Netzwerk besteht aus:

- Input Layer: 784  
- Hidden Layer: 512  
- Output Layer: 10  

---

### Gewichtsmatrizen

Input → Hidden:

$W_1 \in \mathbb{R}^{784 \times 512}$

Hidden → Output:

$W_2 \in \mathbb{R}^{512 \times 10}$

---

### Bias-Terme

$b_1 \in \mathbb{R}^{512}$  
$b_2 \in \mathbb{R}^{10}$

Bias ist ein Grundwert, der zusätzlich zur gewichteten Summe addiert wird.

---

## 3. Forward Pass

### Hidden Layer

Lineare Kombination:

$z_1 = xW_1 + b_1$

Aktivierung:

$a_1 = \mathrm{ReLU}(z_1)$

---

### ReLU-Funktion

$\mathrm{ReLU}(x) = \max(0,x)$

Negative Werte werden entfernt:

$[0,-1,-2,3,1] \mapsto [0,0,0,3,1]$

ReLU dient als Zwischenschritt zwischen den Layern.

---

### Output Layer

$z_2 = a_1W_2 + b_2$

$z_2$ sind die rohen Ausgaben (Logits).

---

## 4. Softmax-Funktion

Softmax wandelt die Output-Werte in Wahrscheinlichkeiten um:

$\hat y_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$

Eigenschaften:

- $\hat y_i > 0$
- $\sum_{i=1}^{10} \hat y_i = 1$

Der größte Wert entspricht der vorhergesagten Klasse.

---

## 5. Loss Function (Cross Entropy)

Die Loss-Funktion misst die Differenz zwischen Vorhersage und Label:

$L = -\sum_{i=1}^{10} y_i \log(\hat y_i)$

Dabei gilt:

- $y$ ist das echte Label (True/False, One-Hot)
- $\hat y$ ist die geschätzte Wahrscheinlichkeit

---

# Backpropagation und Gradient Descent

Training bedeutet, Gewichte und Bias so anzupassen,
dass der Loss minimal wird.

---

## 6. Trainingsloop (Epoche)

Für jedes Trainingsbeispiel $(x,y)$:

1. Forward Pass  
2. Loss berechnen  
3. Backpropagation  
4. Gewichte updaten  

Eine Epoche bedeutet: einmal durch alle Daten laufen.

---

## 7. Gewichtsaktualisierung

Update-Regel:

$w_{\text{neu}} = w_{\text{alt}} - \eta \frac{\partial E}{\partial w}$

Dabei:

- $\eta$ = Lernrate  
- $\frac{\partial E}{\partial w}$ = Gradient der Loss-Funktion

---

## 8. Definition des Gradienten

$\frac{\partial E}{\partial w}
= \lim_{\Delta w \to 0} \frac{\Delta E}{\Delta w}$

Der Gradient gibt an, wie stark sich der Fehler ändert,
wenn man ein Gewicht leicht verändert.

---

## 9. Backpropagation Prinzip

Der Fehler wird vom Output zurück durchs Netz gerechnet,
um Gewichte und Bias anzupassen.

Kettenregel:

$\frac{\partial L}{\partial W}
= \frac{\partial L}{\partial a}
\cdot \frac{\partial a}{\partial z}
\cdot \frac{\partial z}{\partial W}$

---

## 10. Fehlerterm und Updates

Gewichtsänderung:

$\Delta w_{ij} = -\eta \,\delta_j \, o_i$

Bias-Änderung:

$b_{j,\text{neu}} = b_j - \eta \,\delta_j$

Dabei:

- $\delta$ = Fehleranteil  
- $o$ = Output des vorherigen Neurons  

---

## 11. Optimierungs-Intuition

Training kann als Minimierung einer Loss-Landschaft verstanden werden.

Die Parameter bilden einen Parameterraum:

$\theta = \{W_1,W_2,b_1,b_2\}$

Gradient Descent bewegt sich Schritt für Schritt Richtung Minimum:

$\theta \leftarrow \theta - \eta \nabla_\theta L$

---

## 12. Gesamtablauf

Forward:

$x \rightarrow z_1 \rightarrow a_1 \rightarrow z_2 \rightarrow \hat y \rightarrow L$

Backward:

$\delta \rightarrow \nabla W \rightarrow \text{Update}$



![Image](https://github.com/user-attachments/assets/9a4c7b4b-63ff-448c-9d11-3a69f83d2d7b)

![Image (1)](https://github.com/user-attachments/assets/4c082e2f-d6de-4af7-bcf6-d071fce51579)

![Image (2)](https://github.com/user-attachments/assets/768a4d3a-0a6e-420c-bb6a-d73dc8708e42)

![Image (3)](https://github.com/user-attachments/assets/fc3e7202-943e-4a23-bfcf-9b0953c920a0)

![Image (4)](https://github.com/user-attachments/assets/0279f3d2-2c6a-4e66-b2a6-60a848d65a62)
