// cuda_model_multigpu.cu
// ===========================================================================
// Phase 3 EXTENSION: Multi-GPU CUDA.
//
// Same workload as cuda_model.cu, but distributed across two GPUs (developed
// and tested on two RTX 2080 Ti cards connected by NVLink).
//
// Parallelization strategy: DATA PARALLEL.
//   - The 284,807 samples are split roughly in half between the two devices.
//   - Each device runs the same kernel on its half of X / y and produces a
//     partial gradient (a 29-element vector).
//   - The host sums the two partial gradients and updates the weight vector.
//   - The new weights are pushed back to both devices for the next epoch.
//
// Why this is interesting: the kernel is unchanged from the single-GPU version.
// The complexity lives in the orchestration around it - device selection,
// per-device allocations, two synchronization points per epoch, gradient
// aggregation. That orchestration cost is exactly what frameworks like
// Horovod / NCCL exist to hide; doing it by hand shows what they're hiding.
//
// Note on peer access: we try to enable NVLink peer access between the two
// devices, but the current code path still aggregates gradients via the
// host. Switching to a direct device-to-device gradient sum (e.g. via
// cudaMemcpyPeer or NCCL all-reduce) would be the natural next step.
//
// Build:  nvcc -arch=sm_75 cuda_model_multigpu.cu -o cuda_model_multigpu
// Run:    ./cuda_model_multigpu     (requires >= 2 CUDA devices)
// ===========================================================================
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "csv_loader.hpp"

using namespace std;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                 << " - " << cudaGetErrorString(_err) << endl;                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                     \
    } while (0)

// Identical kernel to cuda_model.cu. The multi-GPU work happens entirely on
// the host side: each device runs THIS kernel on its slice of the data.
__global__ void logistic_regression_kernel(const float* X,
                                           const float* y,
                                           const float* weights,
                                           float*       gradients,
                                           int          samples,
                                           int          features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= samples) return;

    float z = 0.0f;
    for (int j = 0; j < features; j++) {
        z += X[i * features + j] * weights[j];
    }
    float prediction = 1.0f / (1.0f + expf(-z));
    float error      = prediction - y[i];
    for (int j = 0; j < features; j++) {
        atomicAdd(&gradients[j], error * X[i * features + j]);
    }
}

int main() {
    cout << "--- Phase 3 Extension: Multi-GPU CUDA C/C++ ---" << endl;

    // ---- 1. Verify we have at least two GPUs -----------------------------
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        cerr << "ERROR: this binary requires at least 2 CUDA devices. "
             << "Found " << device_count << "." << endl;
        return EXIT_FAILURE;
    }
    cout << "-> Detected " << device_count << " CUDA devices" << endl;
    for (int d = 0; d < 2; d++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        cout << "   Device " << d << ": " << prop.name << endl;
    }

    // ---- 2. Enable peer access if available (NVLink path) ----------------
    int can_access_01 = 0;
    int can_access_10 = 0;
    cudaDeviceCanAccessPeer(&can_access_01, 0, 1);
    cudaDeviceCanAccessPeer(&can_access_10, 1, 0);
    if (can_access_01 && can_access_10) {
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceEnablePeerAccess(1, 0);
        CUDA_CHECK(cudaSetDevice(1));
        cudaDeviceEnablePeerAccess(0, 0);
        cout << "-> Peer access enabled between GPU 0 and GPU 1 (NVLink)" << endl;
    } else {
        cout << "-> Peer access NOT available; using host-mediated transfer"
             << endl;
    }

    const int   num_samples  = 284807;
    const int   num_features = 29;
    const int   epochs       = 100;
    const float lr           = 0.01f;

    // ---- 3. Split the dataset roughly in half ----------------------------
    const int samples_d0 = num_samples / 2;            // 142,403
    const int samples_d1 = num_samples - samples_d0;   // 142,404

    // Host-side staging buffers for CSV load, weight aggregation, and
    // accuracy evaluation.
    vector<float> X_host(num_samples * num_features, 0.0f);
    vector<float> y_host(num_samples, 0.0f);
    vector<float> weights_host(num_features, 0.0f);

    // ---- 4. CSV load (host) ----------------------------------------------
    cout << "Loading creditcard.csv..." << endl;
    int parse_errors = 0;
    int rows_loaded  = csv_loader::load_creditcard_csv(
        "creditcard.csv", X_host.data(), y_host.data(),
        num_samples, num_features, parse_errors);

    if (rows_loaded == 0) {
        cerr << "ERROR: no rows parsed. Is creditcard.csv in the working "
                "directory?" << endl;
        return EXIT_FAILURE;
    }
    cout << "-> Loaded " << rows_loaded << " rows" << endl;
    if (parse_errors > 0) {
        cerr << "WARNING: " << parse_errors << " parse errors" << endl;
    }
    csv_loader::print_label_stats(y_host.data(), rows_loaded);

    // ---- 5. Allocate device buffers and seed with the data slice ---------
    // Each GPU owns: its half of X, its half of y, a full copy of weights,
    // and a full-size gradient accumulator.
    float *X_d0, *y_d0, *w_d0, *g_d0;
    float *X_d1, *y_d1, *w_d1, *g_d1;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&X_d0, samples_d0 * num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d0, samples_d0 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w_d0, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d0, num_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(X_d0, X_host.data(),
                          samples_d0 * num_features * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d0, y_host.data(),
                          samples_d0 * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&X_d1, samples_d1 * num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d1, samples_d1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w_d1, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_d1, num_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(X_d1,
                          X_host.data() + samples_d0 * num_features,
                          samples_d1 * num_features * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d1,
                          y_host.data() + samples_d0,
                          samples_d1 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- 6. Training loop -------------------------------------------------
    // Per epoch:
    //   (a) push current weights to both devices, zero both gradient buffers
    //   (b) launch kernels concurrently on both devices
    //   (c) wait for both to finish
    //   (d) pull both partial gradients back to host, sum them, update weights
    cout << "Training (multi-GPU, dataset split " << samples_d0
         << " / " << samples_d1 << ")..." << endl;
    auto start = chrono::high_resolution_clock::now();

    const int tpb       = 256;
    const int blocks_d0 = (samples_d0 + tpb - 1) / tpb;
    const int blocks_d1 = (samples_d1 + tpb - 1) / tpb;

    vector<float> g_partial_d0(num_features);
    vector<float> g_partial_d1(num_features);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // (a) Push weights to both devices, zero both gradient buffers.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(w_d0, weights_host.data(),
                              num_features * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(g_d0, 0, num_features * sizeof(float)));

        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(w_d1, weights_host.data(),
                              num_features * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(g_d1, 0, num_features * sizeof(float)));

        // (b) Launch on both devices concurrently. Each launch is async, so
        //     the second launch begins as soon as it is queued; the two
        //     kernels actually overlap on the wire.
        CUDA_CHECK(cudaSetDevice(0));
        logistic_regression_kernel<<<blocks_d0, tpb>>>(
            X_d0, y_d0, w_d0, g_d0, samples_d0, num_features);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaSetDevice(1));
        logistic_regression_kernel<<<blocks_d1, tpb>>>(
            X_d1, y_d1, w_d1, g_d1, samples_d1, num_features);
        CUDA_CHECK(cudaGetLastError());

        // (c) Wait for both to finish before we trust the gradients.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceSynchronize());

        // (d) Pull partial gradients back to host and combine.
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(g_partial_d0.data(), g_d0,
                              num_features * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(g_partial_d1.data(), g_d1,
                              num_features * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Sum partial gradients and update weights on host. Only 29 ops -
        // host overhead is negligible compared to the kernel.
        for (int j = 0; j < num_features; j++) {
            float total_grad = g_partial_d0[j] + g_partial_d1[j];
            weights_host[j] -= lr * (total_grad / num_samples);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "-> Multi-GPU training finished in "
         << chrono::duration<float>(end - start).count()
         << " seconds." << endl;

    // ---- 7. Evaluate accuracy on the full training set (host-side) -------
    cout << "\n--- RESULTS ---" << endl;
    cout << "Calculating accuracy..." << endl;
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        float z = 0.0f;
        for (int j = 0; j < num_features; j++) {
            z += X_host[i * num_features + j] * weights_host[j];
        }
        float pred = (1.0f / (1.0f + expf(-z))) >= 0.5f ? 1.0f : 0.0f;
        if (pred == y_host[i]) correct++;
    }
    cout << "Accuracy: "
         << static_cast<float>(correct) / num_samples << endl;

    // ---- 8. Cleanup -------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(X_d0));
    CUDA_CHECK(cudaFree(y_d0));
    CUDA_CHECK(cudaFree(w_d0));
    CUDA_CHECK(cudaFree(g_d0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(X_d1));
    CUDA_CHECK(cudaFree(y_d1));
    CUDA_CHECK(cudaFree(w_d1));
    CUDA_CHECK(cudaFree(g_d1));
    return 0;
}
