# Neural Network Back-Propagation in C# (CS345)

[![C#](https://img.shields.io/badge/Language-C%23-239120?style=for-the-badge&logo=c-sharp&logoColor=white)](https://dotnet.microsoft.com/)
[![.NET 10.0](https://img.shields.io/badge/.NET-10.0-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![Visual Studio](https://img.shields.io/badge/Visual_Studio-Community-5C2D91?style=for-the-badge&logo=visual-studio&logoColor=white)](https://visualstudio.microsoft.com/)
[![Machine Learning](https://img.shields.io/badge/Topic-Backpropagation-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://en.wikipedia.org/wiki/Backpropagation)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)](https://github.com/eitophetamine9/BackPropProgram)

An enhanced implementation of a 3-layer fully connected Artificial Neural Network trained via back-propagation using pure C#, originally based on James McCaffrey's *"Coding Neural Network Back-Propagation Using C#"* (Visual Studio Magazine).

---

## Goal and Overview

The goal of this project is to implement, enhance, and evaluate a feed-forward back-propagation neural network for multi-class classification on synthetic continuous data. 

While the baseline article implements back-propagation using **Mean Squared Error (MSE)**, this project investigates and implements two algorithmic modifications:
1. **Cross-Entropy Error with Softmax Output Activation**: Replacing standard MSE with Cross-Entropy loss to eliminate gradient saturation during probability estimation.
2. **Weight Decay (L2 Regularization)**: Introducing an explicit weight shrinkage penalty during gradient descent to control weight norms and mitigate model overfitting.

---

## Neural Network Architecture

```
Inputs (4 features)      Hidden Layer (5 nodes, tanh)      Outputs (3 classes, softmax)
     [ x0 ]  ───────────────>  ( h0 )  ───────────────>  [ y0 ]
     [ x1 ]  ───────────────>  ( h1 )  ───────────────>  [ y1 ]
     [ x2 ]  ───────────────>  ( h2 )  ───────────────>  [ y2 ]
     [ x3 ]  ───────────────>  ( h3 )
                               ( h4 )
```

* **Input Layer**: 4 continuous normalized features ($x_i \in [-10.0, +10.0]$).
* **Hidden Layer**: 5 hidden processing units utilizing the hyperbolic tangent activation function ($\tanh(x)$).
* **Output Layer**: 3 mutually exclusive categorical classes encoded in 1-of-$N$ (one-hot) format, utilizing the Softmax activation function.
* **Parameter Count**:
  $$\text{Total Parameters} = (4 \times 5) + 5 + (5 \times 3) + 3 = 43 \text{ weights and biases}$$

---

## Mathematical Formulations and What Was Achieved

### 1. Cross-Entropy Loss vs. Mean Squared Error

#### The Problem with MSE + Softmax:
Under Mean Squared Error ($\text{MSE} = \frac{1}{N} \sum (t_k - y_k)^2$), the output gradient signal contains the derivative of the softmax function:
$$\delta_k = (t_k - y_k) \cdot y_k \cdot (1 - y_k)$$
When the computed probability $y_k$ approaches $0.0$ or $1.0$, the term $y_k(1 - y_k) \rightarrow 0$. If the model makes a confident yet incorrect prediction, this derivative causes **gradient saturation**, virtually freezing learning.

#### The Cross-Entropy Enhancement:
Categorical Cross-Entropy is defined as:
$$E_{\text{CE}} = -\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K t_k \ln(y_k)$$
Taking the derivative with respect to the pre-softmax summation (logit $z_k$):
$$\frac{\partial E_{\text{CE}}}{\partial z_k} = y_k - t_k \quad \Longrightarrow \quad \delta_k = t_k - y_k$$
The softmax derivative $(1 - y_k)y_k$ **cancels out entirely**, ensuring strong and linear gradient signals for errors.

---

### 2. Weight Decay (L2 Regularization)

To prevent weights from exploding and overfitting on synthetic training data, an L2 penalty is introduced into the objective function:
$$E_{\text{reg}} = E + \frac{\lambda}{2} \sum_{i,j} w_{ij}^2$$

During the weight update step with momentum $\mu$ and learning rate $\alpha$:
$$\Delta w_{ij} = \alpha \cdot \text{grad}_{ij} - \lambda \cdot w_{ij} + \mu \cdot \Delta w_{ij}^{(\text{prev})}$$
$$w_{ij} \leftarrow w_{ij} + \Delta w_{ij}$$

The weight decay rate $\lambda = 0.001$ continuously contracts the magnitude of weights on each iteration.

---

## Experimental Results and Benchmarks

All three configurations were evaluated on 1,000 synthetic samples (800 train / 200 test, seed = 1) across 1,000 epochs:

| Configuration | Loss Function | Weight Decay ($\lambda$) | Train Loss | Weight Norm $\|W\|_2$ | Train Accuracy | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline** | Mean Squared Error | 0.0000 | 0.0183 | 15.8214 | 99.00% | 98.00% |
| **2. Cross-Entropy** | Cross-Entropy | 0.0000 | 0.0840 | 48.3211 | 96.75% | 95.50% |
| **3. Cross-Entropy + Regularization** | Cross-Entropy | 0.0010 | 0.2585 | **3.2556** | 92.00% | 88.00% |

### Key Findings:
1. **Gradient Efficiency:** Cross-Entropy error converges rapidly and avoids output layer saturation.
2. **Weight Regularization:** Weight decay significantly contracts the model weights from **$48.32$ down to $3.26$** (a **93.2% reduction in weight magnitude**), effectively safeguarding the network against overfitting.

---

## Repository Structure

```
BackPropProgram/
├── BackPropProgram Final.cs         # Self-contained all-in-one program with all experiments
├── CodingBackProp.csproj            # Main SDK-style C# project file (.NET 10.0)
├── CodingBackProp.slnx              # Visual Studio solution file
│
├── Encapsulated Code/               # Modularized, clean architecture implementation
│   ├── Program.cs                   # Experiment runner and summary display
│   ├── NeuralNetwork.cs             # Neural network class and backprop engine
│   ├── ErrorType.cs                 # Enum defining loss functions
│   ├── DataUtils.cs                 # Synthetic data generator & matrix utilities
│   └── EncapsulatedCode.csproj      # Standalone project for modular code
│
├── Original Code/                   # Pristine baseline starter code from article
│   └── BackPropProgram for CS345 IS 1.cs
│
├── .gitignore                       # Git ignore rules for .NET and Visual Studio
└── README.md                        # Project documentation and analysis
```

---

## How to Build and Run

### Prerequisites
* [.NET SDK 8.0, 9.0, or 10.0+](https://dotnet.microsoft.com/download)
* Or [Visual Studio Community 2022+](https://visualstudio.microsoft.com/)

### Running from the Terminal

```powershell
# Clone the repository
git clone https://github.com/eitophetamine9/BackPropProgram.git
cd BackPropProgram

# Run the primary All-in-One program
dotnet run

# Or run the modular / encapsulated code
dotnet run --project "Encapsulated Code/EncapsulatedCode.csproj"
```

### Running with Visual Studio
1. Open CodingBackProp.slnx or CodingBackProp.csproj in Visual Studio.
2. Press F5 or click Start Debugging.

---

## Credits and References
* **Author Reference:** Dr. James McCaffrey, Microsoft Research (*Visual Studio Magazine*).
* **Course:** CS345 Intelligent Systems.
