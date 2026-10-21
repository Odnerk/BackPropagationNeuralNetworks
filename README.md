# BackPropagation Neural Networks

[![C#](https://img.shields.io/badge/Language-C%23-239120?style=for-the-badge&logo=c-sharp&logoColor=white)](https://dotnet.microsoft.com/)
[![.NET 10.0](https://img.shields.io/badge/.NET-10.0-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)](https://github.com/Odnerk/BackPropagationNeuralNetworks)

A from-scratch implementation of a feed-forward neural network trained with back-propagation in C#. Built for CS345 Intelligent Systems, starting from a baseline MSE implementation and extended to support Cross-Entropy loss and L2 regularization.

---

## Overview

Three training configurations are implemented and compared side-by-side:

| # | Configuration | Loss Function | Weight Decay |
|---|---|---|---|
| 1 | Baseline | Mean Squared Error | None |
| 2 | Cross-Entropy | Cross-Entropy | None |
| 3 | Cross-Entropy + Regularization | Cross-Entropy | L2 ($\lambda = 0.001$) |

---

## Architecture

```
Input (4)     Hidden (5, tanh)     Output (3, softmax)
  x0 ───────────> h0 ─────────────────> y0
  x1 ───────────> h1 ─────────────────> y1
  x2 ───────────> h2 ─────────────────> y2
  x3 ───────────> h3
                  h4
```

- 4 inputs, 5 hidden nodes (tanh), 3 output classes (softmax + one-hot)
- 43 total parameters: $(4 \times 5) + 5 + (5 \times 3) + 3$
- Trained on 1,000 synthetic samples — 800 train / 200 test

---

## Why Cross-Entropy

With MSE, the output gradient is scaled by the softmax derivative:

$$\delta_k = (t_k - y_k) \cdot y_k(1 - y_k)$$

When the network is confidently wrong ($y_k \to 0$ or $1$), this term collapses to zero and learning stalls. Switching to Cross-Entropy removes that factor entirely:

$$\frac{\partial E_{\text{CE}}}{\partial z_k} = y_k - t_k \quad \Longrightarrow \quad \delta_k = t_k - y_k$$

The gradient is now directly proportional to the error, which keeps training stable.

---

## Why Weight Decay

Without regularization, weights can grow large and overfit the training data. Adding an L2 penalty to the loss:

$$E_{\text{reg}} = E + \frac{\lambda}{2} \sum_{i,j} w_{ij}^2$$

translates to a simple shrinkage term in the update rule:

$$w_{ij} \leftarrow w_{ij} + \alpha \cdot \nabla w_{ij} - \lambda \cdot w_{ij} + \mu \cdot \Delta w_{ij}^{(\text{prev})}$$

---

## Results

1,000 epochs, seed = 1:

| Configuration | Train Loss | $\|W\|_2$ | Train Acc | Test Acc |
| :--- | :---: | :---: | :---: | :---: |
| 1. Baseline (MSE) | 0.0183 | 15.82 | 99.00% | 98.00% |
| 2. Cross-Entropy | 0.0840 | 48.32 | 96.75% | 95.50% |
| 3. Cross-Entropy + Decay | 0.2585 | **3.26** | 92.00% | 88.00% |

Weight decay reduced the L2 norm by **93%** (48.32 → 3.26), significantly constraining weight magnitude at the cost of some accuracy.

---

## Structure

```
BackPropagationNeuralNetworks/
├── BackPropProgram.cs               # All-in-one single-file version
├── BackPropProgram.csproj
├── BackPropProgram.slnx
│
├── src/                             # Modular version split across files
│   ├── Program.cs
│   ├── NeuralNetwork.cs
│   ├── DataUtils.cs
│   ├── ErrorType.cs
│   └── BackPropProgram.Modular.csproj
│
└── original/                        # Unmodified starter code from the article
    └── BackPropProgram for CS345 IS 1.cs
```

---

## Running

**Requirements:** [.NET SDK 10.0+](https://dotnet.microsoft.com/download) or [Visual Studio 2022+](https://visualstudio.microsoft.com/)

```powershell
git clone https://github.com/Odnerk/BackPropagationNeuralNetworks.git
cd BackPropagationNeuralNetworks

# Single-file version
dotnet run

# Modular version
dotnet run --project "src/BackPropProgram.Modular.csproj"
```

In Visual Studio: open `BackPropProgram.slnx` and press **F5**.

---

## Credits

Based on Dr. James McCaffrey's *"Coding Neural Network Back-Propagation Using C#"* (Visual Studio Magazine) — extended for CS345 Intelligent Systems.
