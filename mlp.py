"""
 Multi-Layer Perceptron (Neural Network) — Built from Scratch with NumPy

Architecture overview

We implement two classes:

  Layer  — Encapsulates a single fully-connected layer (weights, biases,
           activation, forward pass, backward pass, and parameter update).

  NeuralNetwork — Orchestrates a stack of Layer objects, running the full
                  forward pass, loss computation, backpropagation, and
                  gradient-descent update across all layers.

Mathematical conventions

  m        : number of training samples (columns of the data matrix)
  n_in     : number of input features  (= neurons in the *previous* layer)
  n_out    : number of output features  (= neurons in *this* layer)

  X        : input matrix,  shape (n_in,  m)
  W        : weight matrix, shape (n_out, n_in)
  b        : bias vector,   shape (n_out, 1)
  Z = WX+b : pre-activation,  shape (n_out, m)
  A = g(Z) : post-activation, shape (n_out, m)

"""

import numpy as np



#  Activation Functions

# Each activation is a pair: (forward_fn, derivative_fn).
# The derivative is expressed in terms of Z (the pre-activation), because
# that is what we cache during the forward pass.

def relu(Z: np.ndarray) -> np.ndarray:
    """
    ReLU activation — element-wise max(0, z).

    Parameters

    Z : np.ndarray, shape (n_out, m)
        Pre-activation matrix.

    Returns

    np.ndarray, shape (n_out, m)
        Post-activation: every negative element is clamped to 0.
    """
    # np.maximum broadcasts element-wise: for each z_ij,
    # returns max(0, z_ij).  The result has the same shape as Z.
    return np.maximum(0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU with respect to Z.

    g'(z) = 1 if z > 0, else 0.

    Parameters

    Z : np.ndarray, shape (n_out, m)

    Returns

    np.ndarray, shape (n_out, m)
        Binary mask: 1 where Z > 0, 0 elsewhere.
    """
    # (Z > 0) produces a boolean array; .astype(float) converts
    # True → 1.0, False → 0.0.  Same shape as Z.
    return (Z > 0).astype(float)


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation — element-wise 1 / (1 + exp(-z)).

    Parameters

    Z : np.ndarray, shape (n_out, m)

    Returns

    np.ndarray, shape (n_out, m)
        Each element is in (0, 1).
    """
    # np.clip prevents overflow in exp() for very negative values of Z.
    # Clipping Z to [-500, 500] is safe: sigmoid(-500) ≈ 0, sigmoid(500) ≈ 1.
    Z_safe = np.clip(Z, -500, 500)

    # Element-wise: σ(z) = 1 / (1 + e^{-z})
    return 1.0 / (1.0 + np.exp(-Z_safe))


def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid with respect to Z.

    σ'(z) = σ(z) · (1 − σ(z))

    Parameters

    Z : np.ndarray, shape (n_out, m)

    Returns

    np.ndarray, shape (n_out, m)
    """
    s = sigmoid(Z)         # shape (n_out, m) — reuse the forward function
    return s * (1.0 - s)   # element-wise multiply, same shape


# Map activation names to (forward, derivative) pairs for convenience.
ACTIVATIONS = {
    "relu":    (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
}

#  Layer Class

class Layer:
    """
    A single fully-connected (dense) layer.

    Stores:
      - W  : weight matrix,  shape (n_out, n_in)
      - b  : bias vector,    shape (n_out, 1)
      - Cached forward-pass intermediates (A_prev, Z) for backprop.
      - Computed gradients (dW, db) after backward.

    Parameters

    n_in  : int   — number of input neurons  (previous layer size)
    n_out : int   — number of output neurons  (this layer size)
    activation : str — "relu" or "sigmoid"
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "relu"):
        # Store layer dimensions 
        self.n_in = n_in    # number of neurons feeding into this layer
        self.n_out = n_out  # number of neurons in this layer

        # He Initialization
        # For ReLU: W ~ N(0, sqrt(2 / n_in))
        # For Sigmoid: Xavier variant W ~ N(0, sqrt(1 / n_in))
        # He init keeps the variance of activations stable across layers,
        # preventing vanishing / exploding signals.
        if activation == "relu":
            scale = np.sqrt(2.0 / n_in)   # He scale factor
        else:
            scale = np.sqrt(1.0 / n_in)   # Xavier scale factor

        # W shape: (n_out, n_in) — each row is the weight vector for one neuron.
        # We draw from a standard normal and scale by the init factor.
        self.W = np.random.randn(n_out, n_in) * scale

        # b shape: (n_out, 1) — one bias per neuron, initialized to zero.
        # Biases start at zero because the random W already breaks symmetry.
        self.b = np.zeros((n_out, 1))

        # Activation functions
        self.activation_name = activation
        # Unpack the (forward, derivative) pair from our lookup table.
        self.g, self.g_prime = ACTIVATIONS[activation]

        # ----- Cache (populated during forward, consumed during backward)
        self.A_prev = None   # Input to this layer,   shape (n_in,  m)
        self.Z      = None   # Pre-activation,        shape (n_out, m)

        # Gradients (populated during backward, consumed during update)
        self.dW = None   # ∂L/∂W, shape (n_out, n_in)
        self.db = None   # ∂L/∂b, shape (n_out, 1)

    #  Forward pass

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Compute Z = W · A_prev + b, then A = g(Z).

        Parameters
        
        A_prev : np.ndarray, shape (n_in, m)
            Activations from the previous layer (or the input data X).

        Returns
        
        A : np.ndarray, shape (n_out, m)
            Activations of this layer.

        Detailed matrix arithmetic
        
        np.dot(self.W, A_prev):
            W      is (n_out, n_in)
            A_prev is (n_in,  m)
            Result is (n_out, m)        — standard matrix multiply

        + self.b:
            self.b is (n_out, 1)
            NumPy broadcasts it across all m columns automatically,
            effectively adding the same bias vector to every sample.

        g(Z):
            Element-wise activation; shape is preserved: (n_out, m).
        """
        # Cache input for use in backward pass.
        self.A_prev = A_prev                                   # (n_in, m)

        # Linear transformation: Z = W · A_prev + b
        # np.dot performs matrix multiplication:
        #   (n_out, n_in) · (n_in, m) → (n_out, m)
        # self.b is (n_out, 1) and broadcasts to (n_out, m).
        self.Z = np.dot(self.W, A_prev) + self.b               # (n_out, m)

        # Non-linear activation: A = g(Z)
        A = self.g(self.Z)                                      # (n_out, m)

        return A

    #  Backward pass

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Given dA = ∂L/∂A (gradient of loss w.r.t. this layer's output),
        compute dW, db for this layer and return dA_prev to propagate
        the gradient further back.

        Parameters
        
        dA : np.ndarray, shape (n_out, m)
            Gradient flowing back from the layer above (or the loss).

        Returns
        
        dA_prev : np.ndarray, shape (n_in, m)
            Gradient to pass to the previous layer.

        Mathematical derivation (chain rule)
        
        We need three gradients:

        1) dZ = dA ⊙ g'(Z)
           - dA      is ∂L/∂A                  (n_out, m)
           - g'(Z)   is the activation derivative (n_out, m)
           - ⊙ is element-wise (Hadamard) product
           - Result dZ = ∂L/∂Z                  (n_out, m)

        2) dW = (1/m) · dZ · A_prev^T
           - dZ      is (n_out, m)
           - A_prev^T is (m, n_in)
           - Result dW = ∂L/∂W                 (n_out, n_in)
           - The 1/m averages the gradient across all m samples.

        3) db = (1/m) · Σ_columns dZ
           - We sum dZ along axis=1 (across samples) and keep dims.
           - Result db = ∂L/∂b                  (n_out, 1)

        4) dA_prev = W^T · dZ
           - W^T    is (n_in, n_out)
           - dZ     is (n_out, m)
           - Result is (n_in, m)
           - This "reverses" the forward multiplication, routing the
             error signal back through the weights to the previous layer.
        """
        m = dA.shape[1]  # number of samples

        # Step 1: Compute dZ — element-wise product of incoming gradient
        #         and the activation derivative evaluated at the cached Z.
        #
        #   dA        : (n_out, m)
        #   g'(Z)     : (n_out, m)
        #   dZ = dA * g'(Z)  — element-wise ("*" in NumPy is Hadamard)
        dZ = dA * self.g_prime(self.Z)                           # (n_out, m)

        # Step 2: Weight gradient — outer product averaged over samples.
        #
        #   dZ          : (n_out, m)
        #   A_prev.T    : (m, n_in)
        #   np.dot →    : (n_out, n_in)  ← same shape as W
        #   Divide by m to get the *mean* gradient across the batch.
        self.dW = (1.0 / m) * np.dot(dZ, self.A_prev.T)         # (n_out, n_in)

        # Step 3: Bias gradient — sum across the sample axis.
        #
        #   np.sum(dZ, axis=1, keepdims=True) sums each row across
        #   all m columns, producing shape (n_out, 1).
        #   Divide by m to average.
        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True) # (n_out, 1)

        # Step 4: Propagate the gradient to the previous layer.
        #
        #   W.T     : (n_in, n_out)
        #   dZ      : (n_out, m)
        #   np.dot → (n_in, m)  ← same shape as A_prev
        #
        #   Conceptually: each neuron in the previous layer receives
        #   error contributions from all neurons in this layer, weighted
        #   by the connecting weights (transposed).
        dA_prev = np.dot(self.W.T, dZ)                           # (n_in, m)

        return dA_prev

    #  Parameter update (vanilla gradient descent)
    
    def update(self, learning_rate: float) -> None:
        """
        Apply one step of gradient descent:

            W ← W − α · dW
            b ← b − α · db

        Parameters
        
        learning_rate : float (α)
            Step size.  Typical values: 0.001 – 0.1
        """
        # Subtract the gradient scaled by the learning rate.
        # Shapes are guaranteed to match: dW ↔ W, db ↔ b.
        self.W -= learning_rate * self.dW   # (n_out, n_in)
        self.b -= learning_rate * self.db   # (n_out, 1)

#  Neural Network Class

class NeuralNetwork:
    """
    A sequential stack of Layer objects forming a Multi-Layer Perceptron.

    Parameters
    
    layer_dims : list[int]
        Number of neurons in each layer, *including* the input layer.
        Example: [2, 64, 32, 1] means:
          - 2 input features
          - 64-neuron hidden layer 1
          - 32-neuron hidden layer 2
          - 1 output neuron
    activations : list[str]
        Activation function for each layer *after* the input.
        Length must be len(layer_dims) - 1.
        Example: ["relu", "relu", "sigmoid"]
    """

    def __init__(self, layer_dims: list, activations: list):
        assert len(activations) == len(layer_dims) - 1, (
            "Need exactly one activation per connection between layers."
        )

        # Build the list of Layer objects.
        # Each Layer connects layer_dims[i] → layer_dims[i+1].
        self.layers = []
        for i in range(len(activations)):
            layer = Layer(
                n_in=layer_dims[i],        # neurons in previous layer
                n_out=layer_dims[i + 1],   # neurons in this layer
                activation=activations[i], # activation function name
            )
            self.layers.append(layer)

    #  Full forward pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Push input X through every layer sequentially.

        Parameters
        
        X : np.ndarray, shape (n_features, m)
            Input data.  Each column is one sample.

        Returns

        A : np.ndarray, shape (n_output, m)
            Network output (predictions).
        """
        A = X  # The input serves as A^[0]

        # Pass the activation forward through each layer.
        for layer in self.layers:
            A = layer.forward(A)  # A^[ℓ] = g(W^[ℓ] · A^[ℓ-1] + b^[ℓ])

        return A  # Final layer output: ŷ

    #  Loss computation — Binary Cross-Entropy

    @staticmethod
    def compute_loss(Y_hat: np.ndarray, Y: np.ndarray) -> float:
        """
        Binary Cross-Entropy loss:

            L = -(1/m) Σ [ y·ln(ŷ) + (1-y)·ln(1-ŷ) ]

        Parameters
        
        Y_hat : np.ndarray, shape (1, m)
            Predicted probabilities from the network (output of sigmoid).
        Y : np.ndarray, shape (1, m)
            Ground-truth binary labels (0 or 1).

        Returns
        
        float
            Scalar loss averaged over all m samples.
        """
        m = Y.shape[1]  # number of samples

        # Clip predictions away from 0 and 1 to prevent log(0) = -inf.
        # This small epsilon (1e-15) has negligible effect on the loss value
        # but guarantees numerical stability.
        Y_hat_safe = np.clip(Y_hat, 1e-15, 1 - 1e-15)

        # Element-wise BCE for each sample:
        #   y·ln(ŷ)           — reward for predicting 1 when true label is 1
        #   (1-y)·ln(1-ŷ)     — reward for predicting 0 when true label is 0
        # np.log is the natural logarithm (element-wise).
        # "*" is element-wise multiplication (Hadamard product).
        loss = -(1.0 / m) * np.sum(
            Y * np.log(Y_hat_safe) + (1 - Y) * np.log(1 - Y_hat_safe)
        )

        return float(loss)

    #  Loss gradient — ∂L/∂A^[L]
    
    @staticmethod
    def compute_loss_gradient(Y_hat: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Gradient of BCE w.r.t. the network output A^[L]:

            ∂L/∂A^[L] = -(1/m) · (Y/ŷ  −  (1−Y)/(1−ŷ))

        However, because our output layer uses sigmoid and the loss is BCE,
        we can simplify the combined gradient ∂L/∂Z^[L] = A^[L] − Y.
        We return dA here and let the layer's backward() apply the sigmoid
        derivative to obtain dZ internally — the math works out identically.

        Parameters
        
        Y_hat : np.ndarray, shape (1, m)
        Y     : np.ndarray, shape (1, m)

        Returns
        
        dA : np.ndarray, shape (1, m)
        """
        # Clip to avoid division by zero.
        Y_hat_safe = np.clip(Y_hat, 1e-15, 1 - 1e-15)

        # ∂L/∂A = -(Y / ŷ) + (1 - Y) / (1 - ŷ)
        #
        # Shape: (1, m) — element-wise arithmetic, same shape as Y_hat.
        # The negative sign is already folded into the formula
        # (matching the negative in the BCE definition).
        dA = -(Y / Y_hat_safe) + (1 - Y) / (1 - Y_hat_safe)

        return dA

    #  Full backward pass
    
    def backward(self, Y_hat: np.ndarray, Y: np.ndarray) -> None:
        """
        Run backpropagation through all layers.

        1. Compute the loss gradient w.r.t. the output.
        2. Propagate it backward through each layer (in reverse order).
        Each layer caches its own dW and db.

        Parameters
        
        Y_hat : np.ndarray, shape (1, m) — predictions
        Y     : np.ndarray, shape (1, m) — true labels
        """
        # Start with the gradient at the output of the network.
        dA = self.compute_loss_gradient(Y_hat, Y)  # (1, m)

        # Walk backward through the layers: L, L-1, …, 1
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            # After this call:
            #   layer.dW and layer.db are set (gradients for this layer).
            #   dA is now ∂L/∂A^[ℓ-1], ready for the next layer back.

    #  Parameter update for all layers

    def update(self, learning_rate: float) -> None:
        """
        Apply gradient descent to every layer.

        Parameters
        
        learning_rate : float
        """
        for layer in self.layers:
            layer.update(learning_rate)

    #  Prediction and accuracy
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions.

        Parameters
        
        X : np.ndarray, shape (n_features, m)
        threshold : float — decision boundary (default 0.5)

        Returns
        
        np.ndarray, shape (1, m) — binary predictions (0 or 1)
        """
        Y_hat = self.forward(X)                  # (1, m) probabilities
        return (Y_hat >= threshold).astype(int)   # threshold → binary

    @staticmethod
    def accuracy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        Classification accuracy.

        Parameters
        
        Y_pred : np.ndarray, shape (1, m) — predicted labels
        Y_true : np.ndarray, shape (1, m) — true labels

        Returns
        
        float — fraction of correct predictions
        """
        # (Y_pred == Y_true) produces a boolean array.
        # np.mean converts True→1, False→0 and averages.
        return float(np.mean(Y_pred == Y_true))

#  Synthetic Dataset: Two Concentric Circles (non-linearly separable)

def make_circles(n_samples: int = 1000,
                 noise: float = 0.05,
                 seed: int = 42) -> tuple:
    """
    Generate a 2D dataset of two concentric circles.

    Inner circle → label 0  (radius ≈ 0.5)
    Outer circle → label 1  (radius ≈ 1.0)

    This is a classic non-linearly separable problem: no straight line
    can separate the two classes.

    Parameters
    
    n_samples : int   — total number of points (split evenly)
    noise     : float — Gaussian noise added to coordinates
    seed      : int   — random seed for reproducibility

    Returns
    
    X : np.ndarray, shape (2, n_samples)
        2D coordinates (each column is one point).
    Y : np.ndarray, shape (1, n_samples)
        Binary labels.
    """
    rng = np.random.RandomState(seed)

    n_half = n_samples // 2  # samples per class

    # ---- Outer circle (label = 1) ----
    # Uniformly sample angles in [0, 2π)
    theta_outer = np.linspace(0, 2 * np.pi, n_half, endpoint=False)
    # Parametric circle: x = r·cos(θ), y = r·sin(θ), with r = 1.0
    x_outer = np.cos(theta_outer)                      # (n_half,)
    y_outer = np.sin(theta_outer)                      # (n_half,)

    # ---- Inner circle (label = 0) ----
    theta_inner = np.linspace(0, 2 * np.pi, n_half, endpoint=False)
    # Same parametric form but with r = 0.5
    x_inner = 0.5 * np.cos(theta_inner)               # (n_half,)
    y_inner = 0.5 * np.sin(theta_inner)               # (n_half,)

    # ---- Concatenate and add noise ----
    # Stack into shape (n_samples,) for each coordinate
    x = np.concatenate([x_outer, x_inner])             # (n_samples,)
    y = np.concatenate([y_outer, y_inner])             # (n_samples,)

    # Add small Gaussian noise to make the problem realistic
    x += rng.randn(n_samples) * noise
    y += rng.randn(n_samples) * noise

    # Labels: first n_half are 1 (outer), second n_half are 0 (inner)
    labels = np.concatenate([np.ones(n_half), np.zeros(n_half)])  # (n_samples,)

    # ---- Reshape to network convention: features × samples ----
    # X shape: (2, n_samples) — each column is a 2D point
    X = np.vstack([x, y])                               # (2, n_samples)
    # Y shape: (1, n_samples) — row vector of labels
    Y = labels.reshape(1, -1)                            # (1, n_samples)

    # ---- Shuffle ----
    # Generate a random permutation of sample indices so the two classes
    # are interleaved (important for mini-batch training, good practice
    # even for full-batch).
    perm = rng.permutation(n_samples)
    X = X[:, perm]   # reorder columns (samples)
    Y = Y[:, perm]   # reorder in the same way

    return X, Y

#  Train / Test Split Utility

def train_test_split(X: np.ndarray,
                     Y: np.ndarray,
                     test_ratio: float = 0.2,
                     seed: int = 42) -> tuple:
    """
    Split data into training and test sets.

    The model will ONLY learn from the training set.  The test set is
    held out — the model never sees it during gradient descent — so
    accuracy on the test set proves the network learned a *generalizable*
    pattern rather than memorizing the training points.

    Parameters
    
    X : np.ndarray, shape (n_features, m)
    Y : np.ndarray, shape (1, m)
    test_ratio : float — fraction of data reserved for testing (0.0–1.0)
    seed : int — random seed for reproducible splits

    Returns
    
    X_train, Y_train, X_test, Y_test
    """
    rng = np.random.RandomState(seed)
    m = X.shape[1]                              # total number of samples

    # Randomly shuffle sample indices so the split is unbiased.
    indices = rng.permutation(m)

    # Number of test samples (rounded down).
    n_test = int(m * test_ratio)
    test_idx  = indices[:n_test]                # first n_test indices → test
    train_idx = indices[n_test:]                # remaining indices   → train

    # Slice columns (each column is one sample).
    X_train, Y_train = X[:, train_idx], Y[:, train_idx]
    X_test,  Y_test  = X[:, test_idx],  Y[:, test_idx]

    return X_train, Y_train, X_test, Y_test

#  Training Loop  (now evaluates on a held-out test set)

def train(network: NeuralNetwork,
          X_train: np.ndarray,
          Y_train: np.ndarray,
          X_test: np.ndarray,
          Y_test: np.ndarray,
          epochs: int = 5000,
          learning_rate: float = 0.1,
          print_every: int = 500) -> list:
    """
    Full-batch gradient descent training loop with train/test evaluation.

    The network trains ONLY on (X_train, Y_train).
    Every `print_every` epochs it also evaluates on (X_test, Y_test)
    — data the network has *never* used for weight updates — so we can
    detect overfitting (train acc high, test acc low) vs. genuine
    generalization (both high).

    Parameters

    network       : NeuralNetwork instance
    X_train       : np.ndarray, shape (n_features, m_train) — training inputs
    Y_train       : np.ndarray, shape (1, m_train)          — training labels
    X_test        : np.ndarray, shape (n_features, m_test)  — test inputs
    Y_test        : np.ndarray, shape (1, m_test)           — test labels
    epochs        : int   — number of full passes over the training data
    learning_rate : float — step size α
    print_every   : int   — log metrics every N epochs

    Returns

    history : list[dict] — recorded train/test loss and accuracy
    """
    history = []

    for epoch in range(1, epochs + 1):

        # 1. Forward pass on TRAINING data only
        Y_hat_train = network.forward(X_train)       # shape (1, m_train)

        # 2. Compute training loss
        train_loss = NeuralNetwork.compute_loss(Y_hat_train, Y_train)

        # 3. Backward pass: compute all gradients (from training data)
        network.backward(Y_hat_train, Y_train)

        # 4. Update parameters with gradient descent
        network.update(learning_rate)

        # 5. Logging (train AND test metrics)
        if epoch % print_every == 0 or epoch == 1:

            # Training metrics 
            Y_pred_train = (Y_hat_train >= 0.5).astype(int)
            train_acc = NeuralNetwork.accuracy(Y_pred_train, Y_train)

            # Test metrics (forward pass only — no gradient update)
            # This is the key: the test data was never used for training,
            # so good test accuracy proves the network generalizes.
            Y_hat_test  = network.forward(X_test)            # (1, m_test)
            test_loss   = NeuralNetwork.compute_loss(Y_hat_test, Y_test)
            Y_pred_test = (Y_hat_test >= 0.5).astype(int)
            test_acc    = NeuralNetwork.accuracy(Y_pred_test, Y_test)

            record = {
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss":  test_loss,  "test_acc":  test_acc,
            }
            history.append(record)

            print(f"  Epoch {epoch:>5d}  │  "
                  f"Train Loss: {train_loss:.6f}  Acc: {train_acc*100:6.2f}%  │  "
                  f"Test Loss: {test_loss:.6f}  Acc: {test_acc*100:6.2f}%")

    return history

#  Main — Assemble, train, and verify on UNSEEN data

if __name__ == "__main__":

    # Reproducibility
    np.random.seed(42)

    # 1. Generate dataset
    print("=" * 78)
    print("  Generating Circles dataset (non-linearly separable)")
    print("=" * 78)
    X, Y = make_circles(n_samples=1000, noise=0.08, seed=42)
    print(f"  Total samples:  {X.shape[1]}")
    print(f"  Class balance:  {Y.mean():.1%} positive\n")

    # 2. Train / Test split (80% train, 20% test)
    # The network will NEVER see the test data during training.
    # Good test accuracy = the network learned the underlying pattern.
    # Bad test accuracy  = the network just memorized training points.
    X_train, Y_train, X_test, Y_test = train_test_split(
        X, Y, test_ratio=0.2, seed=42,
    )
    print(f"  Train set:  X{X_train.shape}  Y{Y_train.shape}  "
          f"({X_train.shape[1]} samples)")
    print(f"  Test  set:  X{X_test.shape}   Y{Y_test.shape}   "
          f"({X_test.shape[1]} samples — held out, never trained on)\n")

    # 3. Build the network
    # Architecture: 2 → 64 → 32 → 1
    # Hidden activations: ReLU (prevents vanishing gradients)
    # Output activation:  Sigmoid (produces probability in [0, 1])
    network = NeuralNetwork(
        layer_dims=[2, 64, 32, 1],
        activations=["relu", "relu", "sigmoid"],
    )

    print("Network architecture:")
    for i, layer in enumerate(network.layers, 1):
        print(f"    Layer {i}: {layer.n_in} → {layer.n_out}  "
              f"({layer.activation_name})  "
              f"W{list(layer.W.shape)}  b{list(layer.b.shape)}")
    print()

    # 4. Train (the test set is evaluated but never used for updates)
    print("=" * 78)
    print("Training with full-batch gradient descent")
    print("Test accuracy = generalization proof — model never trains on it")
    print("=" * 78)
    history = train(
        network,
        X_train, Y_train,
        X_test,  Y_test,
        epochs=5000,
        learning_rate=0.1,
        print_every=500,
    )

    # 5. Final evaluation on both sets
    print("\n" + "=" * 78)
    print("  Final Evaluation")
    print("=" * 78)

    # Training set
    Y_pred_train = network.predict(X_train)
    train_acc    = NeuralNetwork.accuracy(Y_pred_train, Y_train)
    train_loss   = NeuralNetwork.compute_loss(network.forward(X_train), Y_train)

    # Test set (the real proof of learning)
    Y_pred_test = network.predict(X_test)
    test_acc    = NeuralNetwork.accuracy(Y_pred_test, Y_test)
    test_loss   = NeuralNetwork.compute_loss(network.forward(X_test), Y_test)

    print(f"  {'':>12s}  {'Loss':>10s}  {'Accuracy':>10s}")
    print(f"  {'─'*36}")
    print(f"  {'Train':>12s}  {train_loss:>10.6f}  {train_acc*100:>9.2f}%")
    print(f"  {'Test':>12s}  {test_loss:>10.6f}  {test_acc*100:>9.2f}%")

    # 6. Verdict
    print(f"\n  {'─'*36}")
    if test_acc >= 0.95:
        print("Network generalizes — high accuracy on test data!")
        print("This proves genuine learning, not memorization.")
    elif test_acc >= 0.80:
        print("Decent generalization, but some overfitting may be present.")
        print("Consider reducing model size or adding regularization.")
    else:
        print("Poor test accuracy — likely overfitting or underfitting.")

    gap = train_acc - test_acc
    if gap > 0.05:
        print(f"\n Overfitting gap: train {train_acc*100:.1f}% vs "
              f"test {test_acc*100:.1f}% (Δ = {gap*100:.1f}%)")
        print("The model memorized some training noise.")
    else:
        print(f"\n Train/test gap: {gap*100:.1f}%  — no significant overfitting.")

    print("=" * 78)

