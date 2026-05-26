// openacc_model.cpp
// ===========================================================================
// Phase 2: OpenACC Directive-Based GPU Acceleration.
//
// This phase sits in the middle of the productivity / performance spectrum.
// We write a serial-looking C++ training loop and add `#pragma acc` directives
// to tell the compiler (nvc++) which loops to offload to the GPU and which
// arrays to keep resident in device memory between epochs. The compiler
// decides the block / grid configuration for us; we never write a kernel
// by hand.
//
// What we are measuring here:
//   - How well a directive-driven compiler can match a hand-written kernel.
//   - The overhead of OpenACC's data-management abstraction.
//
// Algorithm (matched to cpu_baseline.py and cuda_model.cu):
//   - Full-batch gradient descent, 100 epochs, lr=0.01, no regularization,
//     no bias, weights initialized to zero, full 284,807-sample dataset.
//
// Build:  nvc++ -acc -gpu=managed -Minfo=accel \
//               openacc_model.cpp -o openacc_model
// Run:    ./openacc_model
// ===========================================================================
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "csv_loader.hpp"

using namespace std;

// Sigmoid function. Marked `acc routine seq` so the OpenACC compiler is
// allowed to call it from inside a `#pragma acc parallel` region (i.e. from
// GPU code).
#pragma acc routine seq
float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

int main() {
    cout << "--- Phase 2: OpenACC GPU Acceleration ---" << endl;

    const int   num_samples  = 284807;
    const int   num_features = 29;
    const int   epochs       = 100;
    const float lr           = 0.01f;

    // Heap-allocated so they survive the `#pragma acc data` region cleanly.
    float* X         = new float[num_samples * num_features]{0};
    float* y         = new float[num_samples]{0};
    float* weights   = new float[num_features]{0};
    float* gradients = new float[num_features]{0};

    // ---- 1. CSV load ------------------------------------------------------
    cout << "Loading creditcard.csv (skipping header and Time column)..." << endl;
    auto start_io = chrono::high_resolution_clock::now();

    int parse_errors = 0;
    int rows_loaded  = csv_loader::load_creditcard_csv(
        "creditcard.csv", X, y, num_samples, num_features, parse_errors);

    if (rows_loaded == 0) {
        cerr << "ERROR: no rows parsed. Is creditcard.csv in the working "
                "directory?" << endl;
        return EXIT_FAILURE;
    }

    auto end_io = chrono::high_resolution_clock::now();
    cout << "-> Data loaded in "
         << chrono::duration<float>(end_io - start_io).count()
         << " seconds." << endl;
    if (parse_errors > 0) {
        cerr << "WARNING: " << parse_errors
             << " parse errors encountered during CSV load" << endl;
    }
    csv_loader::print_label_stats(y, rows_loaded);

    // ---- 2. Training loop -------------------------------------------------
    // The `acc data` region pins X, y, weights, and gradients on the device
    // for the entire training run. Without it, OpenACC would re-transfer
    // these arrays at every epoch boundary - which would dominate runtime.
    //
    //   copyin   : push to device once, never copy back (X and y are read-only).
    //   copy     : push to device and copy back at the end (weights, the
    //              final trained parameters).
    //   create   : allocate on device only, no transfers needed (gradients
    //              are re-zeroed every epoch).
    cout << "\nTraining (100 epochs, lr=0.01)..." << endl;
    auto start_train = chrono::high_resolution_clock::now();

    #pragma acc data copyin(X[0:num_samples * num_features], y[0:num_samples]) \
                     copy(weights[0:num_features]) \
                     create(gradients[0:num_features])
    {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // (a) Zero the gradient accumulator.
            #pragma acc parallel loop
            for (int j = 0; j < num_features; j++) {
                gradients[j] = 0.0f;
            }

            // (b) Forward + backward pass.
            //     One iteration per sample i. The `reduction(+:gradients[...])`
            //     clause tells the compiler to give each thread a private
            //     copy of the gradient vector and sum them at the end -
            //     this is the directive-based equivalent of atomicAdd.
            #pragma acc parallel loop reduction(+:gradients[0:num_features])
            for (int i = 0; i < num_samples; i++) {
                float z = 0.0f;
                for (int j = 0; j < num_features; j++) {
                    z += X[i * num_features + j] * weights[j];
                }
                float error = sigmoid(z) - y[i];
                for (int j = 0; j < num_features; j++) {
                    gradients[j] += error * X[i * num_features + j];
                }
            }

            // (c) Weight update (averaged over the batch).
            #pragma acc parallel loop
            for (int j = 0; j < num_features; j++) {
                weights[j] -= lr * (gradients[j] / num_samples);
            }
        }
    }

    auto end_train = chrono::high_resolution_clock::now();
    cout << "-> Training finished in "
         << chrono::duration<float>(end_train - start_train).count()
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
        float pred = sigmoid(z) >= 0.5f ? 1.0f : 0.0f;
        if (pred == y[i]) correct++;
    }
    float acc = static_cast<float>(correct) / num_samples;
    cout << "Accuracy: " << acc << endl;

    delete[] X;
    delete[] y;
    delete[] weights;
    delete[] gradients;
    return 0;
}
