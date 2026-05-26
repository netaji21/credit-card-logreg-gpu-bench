# Makefile for the GPU logistic regression benchmark.
#
# Targets:
#   make                      build all GPU binaries (openacc + cuda + multi-GPU)
#   make openacc_model        build the OpenACC binary only
#   make cuda_model           build the single-GPU CUDA binary only
#   make cuda_model_multigpu  build the multi-GPU CUDA binary only
#   make clean                remove built binaries
#
# Requirements:
#   - NVIDIA HPC SDK (provides nvc++) for OpenACC
#   - CUDA Toolkit  (provides nvcc)   for the CUDA targets

# --- Compiler discovery -----------------------------------------------------
NVCXX ?= nvc++
NVCC  ?= nvcc

# --- Build flags ------------------------------------------------------------
# -arch=sm_75 targets RTX 2080 Ti / Turing. Override on the command line for
# newer GPUs, e.g.:   make CUDA_ARCH=sm_86
CUDA_ARCH    ?= sm_75
NVCC_FLAGS   := -O2 -arch=$(CUDA_ARCH) -std=c++14 -Isrc
OPENACC_FLAGS:= -O2 -acc -gpu=managed -Minfo=accel -Isrc

# --- Targets ----------------------------------------------------------------
.PHONY: all clean
all: openacc_model cuda_model cuda_model_multigpu

openacc_model: src/openacc_model.cpp src/csv_loader.hpp
	$(NVCXX) $(OPENACC_FLAGS) src/openacc_model.cpp -o openacc_model

cuda_model: src/cuda_model.cu src/csv_loader.hpp
	$(NVCC) $(NVCC_FLAGS) src/cuda_model.cu -o cuda_model

cuda_model_multigpu: src/cuda_model_multigpu.cu src/csv_loader.hpp
	$(NVCC) $(NVCC_FLAGS) src/cuda_model_multigpu.cu -o cuda_model_multigpu

clean:
	rm -f openacc_model cuda_model cuda_model_multigpu
