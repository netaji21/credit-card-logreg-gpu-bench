"""
cpu_baseline.py
===============
CPU reference implementation for the GPU benchmark.

This file exists to anchor the comparison: it runs the same logistic
regression on the same data with the same hyperparameters as the OpenACC
and CUDA implementations, but on a single CPU thread (via NumPy). The
accuracy this file reports should match the OpenACC and CUDA results to
4 decimal places - if it doesn't, one of the GPU implementations has
diverged from the reference math and the comparison is no longer apples-
to-apples.

Algorithm (matched across all four implementations except RAPIDS):
    - Full-batch gradient descent (NOT stochastic / mini-batch)
    - 100 epochs
    - Learning rate 0.01
    - No regularization
    - No bias / intercept term (weights initialized to zero)
    - 29 features (the non-predictive "Time" column is dropped)
    - Trained and evaluated on the full 284,807-sample dataset

Note: RAPIDS (rapids_model.py) uses cuML's L-BFGS solver internally, which
is a different optimizer. That solver-level difference is one of the
trade-offs the benchmark is measuring.
"""

import time

import numpy as np
import pandas as pd
from scipy.special import expit  # numerically-stable sigmoid


def main() -> None:
    print("--- CPU Baseline: Full-Batch Gradient Descent (NumPy) ---")

    # 1. Load data ----------------------------------------------------------
    print("Loading creditcard.csv...")
    t0 = time.time()
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Time", "Class"]).to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.float32)
    io_time = time.time() - t0
    print(f"-> Data loaded in {io_time:.4f} seconds.  Shape: X={X.shape}, y={y.shape}")

    num_samples, num_features = X.shape

    # 2. Initialize weights to zero (matches C++ implementations exactly) ---
    weights = np.zeros(num_features, dtype=np.float32)

    epochs = 100
    lr = 0.01

    # 3. Training loop - full-batch gradient descent ------------------------
    # For each epoch:
    #   z        = X @ w                 (forward pass: linear scores)
    #   error    = sigmoid(z) - y        (residual on every sample)
    #   gradient = X.T @ error           (sum the per-sample gradients)
    #   w       -= lr * gradient / N     (weight update, averaged over batch)
    print("\nTraining (100 epochs, lr=0.01, full-batch GD, no regularization)...")
    t0 = time.time()

    for _ in range(epochs):
        z = X @ weights
        error = expit(z) - y
        gradient = X.T @ error
        weights -= lr * (gradient / num_samples)

    train_time = time.time() - t0
    print(f"-> CPU training finished in {train_time:.4f} seconds.")

    # 4. Evaluate on the FULL dataset (matches GPU evaluation protocol) -----
    print("\n--- RESULTS ---")
    z = X @ weights
    preds = (expit(z) >= 0.5).astype(np.float32)
    correct = int((preds == y).sum())
    acc = correct / num_samples
    print(f"Accuracy: {acc:.4f}  ({correct} / {num_samples} correct)")
    print(f"Total CPU time (load + train): {io_time + train_time:.4f} seconds")


if __name__ == "__main__":
    main()
