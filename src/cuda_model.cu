// cuda_model.cu
// ===========================================================================
// Phase 3: Manual CUDA C/C++ Kernel.
//
// This phase sits at the "raw performance" end of the spectrum. We write the
// kernel by hand: explicit thread / block configuration, explicit atomicAdd
// for gradient aggregation, explicit Unified Memory for host <-> device data
// sharing. Nothing is hidden.
//
// What we are measuring here:
//   - Best-case throughput achievable for this workload with a straightforward
//     hand-written kernel.
//   - The development cost (lines of code, boilerplate) that buys that
//     throughput relative to RAPIDS (Phase 1) and OpenACC (Phase 2).
//
// Algorithm (matched to cpu_baseline.py and openacc_model.cpp):
//   - Full-batch gradient descent, 100 epochs, lr=0.01, no regularization,
//     no bias, weights initialized to zero, full 284,807-sample dataset.
//
// Memory model:
//   cudaMallocManaged returns Unified-Memory pointers usable from BOTH host
//   code (CSV load, accuracy evaluation, weight update) and device code (the
//   kernel). The driver migrates pages on demand. This is the simplest
//   correct memory strategy for a learning-focused implementation; a
//   production version would use explicit cudaMemcpyAsync + streams.
//
// Build:  nvcc -arch=sm_75 cuda_model.cu -o cuda_model
// Run:    ./cuda_model
// ===========================================================================
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include "csv_loader.hpp"

using namespace std;

// Wrap a CUDA runtime call and abort cleanly on error. Without this every
// runtime call would need a 5-line if-block, which would drown out the
// actual logic.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                 << " - " << cudaGetErrorString(_err) << endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

// Per-sample forward + backward pass.
//
// Launch configuration: one thread per training sample. Thread i computes
//   z_i        = X_i . weights           (forward)
//   error_i    = sigmoid(z_i) - y_i      (residual)
//   gradient  += error_i * X_i           (backward, accumulated into shared
//                                          gradient vector via atomicAdd)
//
// atomicAdd is required because every thread writes into the same 29-element
// gradient vector. With ~285k threads racing on 29 slots, contention IS the
// bottleneck of this kernel - a shared-memory reduction would be the next
// optimization step.
__global__ void logistic_regression_kernel(const float* X,
                                           const float* y,
                                           const float* weights,
                                           float*       gradients,
                                           int          samples,
                                           int          features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= samples) return;

    // Forward pass: z = X_i . weights
    float z = 0.0f;
    for (int j = 0; j < features; j++) {
        z += X[i * features + j] * weights[j];
    }

    // Residual.
    float prediction = 1.0f / (1.0f + expf(-z));
    float error      = prediction - y[i];

    // Backward pass: accumulate this sample's gradient contribution.
    for (int j = 0; j < features; j++) {
        atomicAdd(&gradients[j], error * X[i * features + j]);
    }
}

int main() {
    cout << "--- Phase 3: CUDA C/C++ Manual Kernel ---" << endl;

    const int   num_samples  = 284807;
    const int   num_features = 29;
    const int   epochs       = 100;
    const float lr           = 0.01f;

    float *X, *y, *weights, *gradients;

    // Unified Memory: single pointer usable from host and device.
    CUDA_CHECK(cudaMallocManaged(&X,         num_samples * num_features * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&y,         num_samples * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&weights,   num_features * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&gradients, num_features * sizeof(float)));

    // cudaMallocManaged does NOT guarantee zero-initialized memory. Touch
    // every element from the host so first-touch happens here, not in the
    // middle of training.
    for (int i = 0; i < num_samples * num_features; i++) X[i] = 0.0f;
    for (int i = 0; i < num_samples;                i++) y[i] = 0.0f;
    for (int i = 0; i < num_features;               i++) weights[i] = 0.0f;

    // ---- 1. CSV load ------------------------------------------------------
    cout << "Loading creditcard.csv..." << endl;
    int parse_errors = 0;
    int rows_loaded  = csv_loader::load_creditcard_csv(
        "creditcard.csv", X, y, num_samples, num_features, parse_errors);

    if (rows_loaded == 0) {
        cerr << "ERROR: no rows parsed. Is creditcard.csv in the working "
                "directory?" << endl;
        return EXIT_FAILURE;
    }
    cout << "-> Loaded " << rows_loaded << " rows" << endl;
    if (parse_errors > 0) {
        cerr << "WARNING: " << parse_errors
             << " parse errors encountered during CSV load" << endl;
    }
    csv_loader::print_label_stats(y, rows_loaded);

    // ---- 2. Training loop -------------------------------------------------
    cout << "Training..." << endl;
    auto start = chrono::high_resolution_clock::now();

    // 256 threads per block is the conventional default for sm_75. The grid
    // is sized so that there is at least one thread per sample (the extra
    // threads in the last block early-return inside the kernel).
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // (a) Zero gradients on device.
        CUDA_CHECK(cudaMemset(gradients, 0, num_features * sizeof(float)));

        // (b) Forward + backward pass on GPU.
        logistic_regression_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            X, y, weights, gradients, num_samples, num_features);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // (c) Weight update on host. Only 29 elements - the migration cost
        //     and host arithmetic are both negligible relative to the kernel.
        for (int j = 0; j < num_features; j++) {
            weights[j] -= lr * (gradients[j] / num_samples);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "-> CUDA training finished in "
         << chrono::duration<float>(end - start).count()
         << " seconds." << endl;

    // ---- 3. Evaluate accuracy on the full training set -------------------
    cout << "\n--- RESULTS ---" << endl;
    cout << "Calculating accuracy..." << endl;
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        float z = 0.0f;
        for (int j = 0; j < num_features; j++) {
            z += X[i * num_features + j] * weights[j];
        }
        float pred = (1.0f / (1.0f + expf(-z))) >= 0.5f ? 1.0f : 0.0f;
        if (pred == y[i]) correct++;
    }
    float acc = static_cast<float>(correct) / num_samples;
    cout << "Accuracy: " << acc << endl;

    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(gradients));
    return 0;
}
