# Neural Network from Scratch

A fully functioning **Multi-Layer Perceptron (MLP)** built entirely from scratch using only **NumPy** — no TensorFlow, no PyTorch, no Keras. Every matrix operation, every gradient, every weight update is implemented by hand with exhaustive inline comments explaining the math.

## Features

- **Pure NumPy** — zero deep learning framework dependencies
- **Object-Oriented Design** — clean `Layer` and `NeuralNetwork` classes
- **He / Xavier Weight Initialization** — proper variance scaling for stable training
- **ReLU & Sigmoid Activations** — with hand-derived derivatives
- **Binary Cross-Entropy Loss** — with numerically stable clipping
- **Full Backpropagation** — chain rule gradients computed layer by layer
- **Train / Test Split** — proves generalization, not memorization
- **Synthetic Dataset** — non-linearly separable concentric circles

## Architecture

<p align="center">
  <img width="1012" height="604" alt="image" src="https://github.com/user-attachments/assets/dbd51a3b-6f3e-4566-a241-e833963456cc" />

</p>

```
Input (2)  →  Hidden 1 (64, ReLU)  →  Hidden 2 (32, ReLU)  →  Output (1, Sigmoid)
```

| Connection | Weight Shape | Bias Shape |
|-----------|-------------|-----------|
| Input → Hidden 1 | `(64, 2)` | `(64, 1)` |
| Hidden 1 → Hidden 2 | `(32, 64)` | `(32, 1)` |
| Hidden 2 → Output | `(1, 32)` | `(1, 1)` |

## 🔬 Mathematical Foundation

### Forward Propagation

$$Z^{[\ell]} = W^{[\ell]} \cdot A^{[\ell-1]} + b^{[\ell]}$$
$$A^{[\ell]} = g(Z^{[\ell]})$$

### Activation Functions

| Function | Formula | Derivative |
|----------|---------|-----------|
| ReLU | $\max(0, z)$ | $1$ if $z > 0$, else $0$ |
| Sigmoid | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ |

### Binary Cross-Entropy Loss

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

### Backpropagation

$$dZ^{[\ell]} = dA^{[\ell]} \odot g'(Z^{[\ell]})$$
$$dW^{[\ell]} = \frac{1}{m} \, dZ^{[\ell]} \cdot (A^{[\ell-1]})^T$$
$$db^{[\ell]} = \frac{1}{m} \sum dZ^{[\ell]}$$
$$dA^{[\ell-1]} = (W^{[\ell]})^T \cdot dZ^{[\ell]}$$

### Gradient Descent

$$W \leftarrow W - \alpha \cdot dW \qquad b \leftarrow b - \alpha \cdot db$$

## Results

<p align="center">
  <img width="1831" height="891" alt="image" src="https://github.com/user-attachments/assets/70c6f9f3-578e-4cba-851b-51fe482c0a1b" />
</p>

> The network learns a **circular decision boundary** from scratch. Left: training data. Right: unseen test data — proving generalization, not memorization.

##  Quick Start

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib (for visualization only)

```bash
pip install numpy matplotlib
```

### Run

```bash
# Train the network
python mlp.py

# Generate the decision boundary plot
python visualize.py
```

### Expected Output

```
==============================================================================
  Generating Circles dataset (non-linearly separable)
==============================================================================
  Total samples:  1000
  Class balance:  50.0% positive

  Train set:  X(2, 800)  Y(1, 800)  (800 samples)
  Test  set:  X(2, 200)   Y(1, 200)   (200 samples — held out, never trained on)

  Network architecture:
    Layer 1: 2 → 64  (relu)  W[64, 2]  b[64, 1]
    Layer 2: 64 → 32  (relu)  W[32, 64]  b[32, 1]
    Layer 3: 32 → 1  (sigmoid)  W[1, 32]  b[1, 1]

==============================================================================
  Training with full-batch gradient descent
  (Test accuracy = generalization proof — model never trains on it)
==============================================================================
  Epoch     1  │  Train Loss: 0.710374  Acc:  51.50%  │  Test Loss: 0.677815  Acc:  57.00%
  Epoch   500  │  Train Loss: 0.020806  Acc: 100.00%  │  Test Loss: 0.022705  Acc: 100.00%
  Epoch  1000  │  Train Loss: 0.007910  Acc: 100.00%  │  Test Loss: 0.009735  Acc: 100.00%
  ...
  Epoch  5000  │  Train Loss: 0.001257  Acc: 100.00%  │  Test Loss: 0.002874  Acc: 100.00%

==============================================================================
  Final Evaluation
==============================================================================
                    Loss    Accuracy
  ────────────────────────────────────
         Train    0.001257     100.00%
          Test    0.002874     100.00%

   Network generalizes — high accuracy on UNSEEN test data!
   Train/test gap: 0.0%  — no significant overfitting.
```

## Project Structure

```
neural_network/
└── mlp.py          # Complete implementation (Layer, NeuralNetwork, dataset, training)
```

## Code Structure

### Classes

| Class | Description |
|-------|-------------|
| `Layer` | Single fully-connected layer — weights, biases, forward pass, backward pass, gradient descent update |
| `NeuralNetwork` | Sequential stack of `Layer` objects — orchestrates forward/backward passes, loss computation, and parameter updates |

### Key Methods

| Method | What It Does |
|--------|-------------|
| `Layer.forward(A_prev)` | Computes $Z = WA + b$, then $A = g(Z)$. Caches intermediates for backprop |
| `Layer.backward(dA)` | Computes $dW$, $db$ via chain rule. Returns $dA_{\text{prev}}$ to propagate backward |
| `Layer.update(lr)` | Applies $W \leftarrow W - \alpha \cdot dW$ |
| `NeuralNetwork.forward(X)` | Pushes input through all layers sequentially |
| `NeuralNetwork.compute_loss(ŷ, y)` | Binary Cross-Entropy with numerical clipping |
| `NeuralNetwork.backward(ŷ, y)` | Full backpropagation through all layers in reverse |
| `NeuralNetwork.predict(X)` | Thresholds probabilities at 0.5 to produce binary labels |

### Standalone Functions

| Function | What It Does |
|----------|-------------|
| `relu()` / `relu_derivative()` | $\max(0, z)$ activation and its derivative |
| `sigmoid()` / `sigmoid_derivative()` | $\frac{1}{1+e^{-z}}$ activation and its derivative |
| `make_circles()` | Generates non-linearly separable concentric circles dataset |
| `train_test_split()` | Splits data into 80% train / 20% test |
| `train()` | Full training loop with train & test evaluation |

## Educational Notes

### Why Train/Test Split?

Evaluating only on training data proves nothing — the model could simply memorize every point. By holding out 20% of the data that the network **never** trains on, we prove it learned the **underlying pattern** (a circular boundary), not the individual data points.

### Why Concentric Circles?

A linear classifier can only draw a straight line. Circles require a **curved** decision boundary, forcing the network to rely on its non-linear activations (ReLU). This proves the hidden layers are doing meaningful work.

### Why He Initialization?

Random weights drawn from $\mathcal{N}(0, 1)$ would cause activations to explode or vanish across deep layers. He initialization ($\sigma = \sqrt{2/n_{\text{in}}}$) keeps the variance stable, enabling the network to learn from epoch 1.

## Customization

Change the architecture by modifying these two lines:

```python
network = NeuralNetwork(
    layer_dims=[2, 128, 64, 32, 1],                           # Add/change layers
    activations=["relu", "relu", "relu", "sigmoid"],           # One activation per layer
)
```

Tune hyperparameters:

```python
history = train(
    network,
    X_train, Y_train, X_test, Y_test,
    epochs=10000,        # More training iterations
    learning_rate=0.01,  # Smaller steps
    print_every=1000,    # Log frequency
)
```
