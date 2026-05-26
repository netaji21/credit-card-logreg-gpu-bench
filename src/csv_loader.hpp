// csv_loader.hpp
// ---------------------------------------------------------------------------
// Shared CSV loader for the Credit Card Fraud Detection dataset.
//
// All three C++/CUDA implementations (OpenACC, single-GPU CUDA, multi-GPU
// CUDA) need to load the exact same dataset into the exact same layout, so
// the parsing logic lives here as a header-only helper to keep the three
// driver files in sync. If the parser changes, it changes once.
//
// Layout produced:
//   X : row-major float array, shape [num_samples, 29]
//       - The "Time" column is dropped (non-predictive).
//       - Columns V1..V28 and Amount are kept, in that order.
//   y : float array, shape [num_samples], values in {0.0f, 1.0f}.
//
// The dataset wraps the Class column in double quotes ("0", "1"), which
// would otherwise cause std::stoi to throw. clean_cell() strips those.
// ---------------------------------------------------------------------------
#pragma once

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace csv_loader {

// Strip surrounding double-quotes and whitespace from a CSV cell.
inline std::string clean_cell(const std::string& s) {
    size_t start = 0;
    size_t end   = s.size();
    while (start < end &&
           (s[start] == '"' || std::isspace(static_cast<unsigned char>(s[start])))) {
        start++;
    }
    while (end > start &&
           (s[end - 1] == '"' || std::isspace(static_cast<unsigned char>(s[end - 1])))) {
        end--;
    }
    return s.substr(start, end - start);
}

// Load creditcard.csv into the caller-provided X and y buffers.
//
// Returns the number of rows actually parsed. Writes the number of cells that
// failed to parse to `parse_errors` (typically 0 for a clean dataset).
//
// Pre-conditions:
//   - X has space for at least num_samples * num_features floats.
//   - y has space for at least num_samples floats.
//   - num_features == 29 (the loader assumes the Time-dropped layout).
inline int load_creditcard_csv(const std::string& filename,
                               float*             X,
                               float*             y,
                               int                num_samples,
                               int                num_features,
                               int&               parse_errors) {
    parse_errors = 0;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: could not open " << filename << std::endl;
        return 0;
    }

    std::string line;
    std::getline(file, line);  // skip header row

    int rows_loaded = 0;
    for (int row = 0; row < num_samples; row++) {
        if (!std::getline(file, line)) break;

        std::stringstream ss(line);
        std::string       cell;
        int               col_idx  = 0;
        int               feat_idx = 0;

        while (std::getline(ss, cell, ',')) {
            std::string c = clean_cell(cell);
            try {
                if (col_idx == 0) {
                    // Time column: skip.
                } else if (col_idx <= 29) {
                    // V1..V28 and Amount.
                    X[row * num_features + feat_idx] = std::stof(c);
                    feat_idx++;
                } else if (col_idx == 30) {
                    // Class column (binary fraud label).
                    y[row] = static_cast<float>(std::stoi(c));
                }
            } catch (const std::exception&) {
                parse_errors++;
            }
            col_idx++;
        }
        rows_loaded++;
    }
    return rows_loaded;
}

// Print a quick sanity-check line summarizing the parsed labels. The Credit
// Card Fraud dataset is extremely imbalanced - the expected positive rate is
// ~0.00173 (492 / 284,807). A value far from that is a strong signal that the
// CSV parser is reading the wrong column.
inline void print_label_stats(const float* y, int rows_loaded) {
    int positives = 0;
    for (int i = 0; i < rows_loaded; i++) {
        if (y[i] > 0.5f) positives++;
    }
    std::cout << "-> Positive label rate: "
              << static_cast<float>(positives) / rows_loaded
              << " (" << positives << " / " << rows_loaded << ")" << std::endl;
}

}  // namespace csv_loader
