"""
rapids_model.py
===============
Phase 1: High-Level GPU Acceleration via NVIDIA RAPIDS.

This phase represents the "developer productivity" end of the spectrum:
the user writes essentially the same scikit-learn-style code they would
write on the CPU, and the RAPIDS libraries (cuDF for the DataFrame,
cuML for the model) push the data and the math onto the GPU under the
hood.

What we are measuring here:
    - Wall-clock time for direct-to-GPU data load (cuDF.read_csv).
    - Wall-clock time for model.fit() on the GPU.
    - Final accuracy on the full dataset.

Hyperparameter matching (vs. Phases 2/3):
    - max_iter=100         matches the 100 GD epochs in C++.
    - penalty='none'       matches "no regularization" in C++.
    - fit_intercept=False  matches "no bias term" in C++.

Caveat: cuML's LogisticRegression uses an L-BFGS quasi-Newton solver
internally rather than vanilla gradient descent. This is a deliberate
part of what Phase 1 measures - the cost of "library convenience" vs.
"hand-written kernel" includes the optimizer the library happens to
pick for you.
"""

import time

import cudf
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score


def main() -> None:
    print("--- Phase 1: RAPIDS GPU Acceleration ---")

    # 1. Load directly into GPU memory.
    #    cudf.read_csv lands the parsed columns straight on the device,
    #    skipping the CPU DataFrame -> GPU copy that pandas would force.
    print("Loading creditcard.csv into GPU memory...")
    start_io = time.time()
    df = cudf.read_csv("creditcard.csv")
    io_time = time.time() - start_io
    print(f"-> Data loaded in {io_time:.4f} seconds.")

    # 2. Prepare data. Drop "Time" to match the 29-feature layout the
    #    OpenACC and CUDA C/C++ implementations use.
    X = df.drop(["Time", "Class"], axis=1)
    y = df["Class"].astype("float32")

    # 3. Train model.
    print("\nTraining Logistic Regression (100 iter, no penalty, no intercept)...")
    model = LogisticRegression(
        max_iter=100,
        penalty="none",
        fit_intercept=False,
    )

    start_train = time.time()
    model.fit(X, y)
    train_time = time.time() - start_train
    print(f"-> Model trained in {train_time:.4f} seconds.")

    # 4. Evaluate on the FULL dataset (matches OpenACC/CUDA protocol).
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Total GPU compute time (load + train): {io_time + train_time:.4f} seconds")


if __name__ == "__main__":
    main()
