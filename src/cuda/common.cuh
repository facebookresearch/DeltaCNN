// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <vector>
#include <torch/extension.h>

enum PaddingMode {zeros, repeat, mirror};

struct ConvConfig {
    uint16_t kernel_size[2];
    uint16_t stride[2];
    uint16_t dilation[2];
    uint16_t padding[4];
    PaddingMode padding_mode;
    uint16_t groups;
    bool sub_tile_sparsity;
    bool set_sparse_zero;
};

struct ImageDimension{
    uint16_t h;
    uint32_t w;
    uint16_t c;
};

struct Dimensions {
    uint16_t batch_size;
    ImageDimension in;
    ImageDimension out;
};

// #define ENABLE_METRICS



struct DCMetrics {
    // these values are only tracked inside convolutional kernels
    uint64_t n_active_tiles;
    uint64_t n_tiles;
    uint64_t n_active_inputs;
    uint64_t n_inputs;
    uint64_t n_tiles_sparse_mode;
    uint64_t n_tiles_dense_mode;
    uint64_t n_active_flops;
    uint64_t n_theoretical_flops;
    uint64_t n_dense_flops;

    // these values are tracked in all layers
    static const bool track_filter_reads = true;
    uint64_t n_vals_read;
    uint64_t n_vals_read_dense;
    uint64_t n_vals_written;
    uint64_t n_vals_written_dense;

    static const uint64_t histogram_samples = 128;
    // histogram is only tracked inside convolutional kernels
    uint64_t active_input_histogram[histogram_samples];

    static const uint64_t n_samples_total = histogram_samples + 13;
};

#ifdef ENABLE_METRICS
struct DCMetrics_ptrs {
    // these values are only tracked inside convolutional kernels
    uint64_t *n_active_tiles;
    uint64_t *n_tiles;
    uint64_t *n_active_inputs;
    uint64_t *n_inputs;
    uint64_t *n_tiles_sparse_mode;
    uint64_t *n_tiles_dense_mode;
    uint64_t *n_active_flops;
    uint64_t *n_theoretical_flops;
    uint64_t *n_dense_flops;

    // these values are tracked in all layers
    static const bool track_filter_reads = true;
    uint64_t *n_vals_read;
    uint64_t *n_vals_read_dense;
    uint64_t *n_vals_written;
    uint64_t *n_vals_written_dense;

    static const uint64_t histogram_samples = 128;
    // histogram is only tracked inside convolutional kernels
    uint64_t *active_input_histogram;

    static const uint64_t n_samples_total = histogram_samples + 13;
};

static DCMetrics *d_metrics_ptr_copy;
static DCMetrics h_metrics;
#endif

bool init_performance_metrics();
void reset_performance_metrics();
std::vector<torch::Tensor> retrieve_metrics();
void copy_performance_metrics_to_gpu(DCMetrics*& d);

inline static void HandleError(cudaError_t err,
							   const char *file,
							   int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			   file, line);
		throw std::exception();
	}
}
// #ifdef _DEBUG || NDEBUG || DEBUG
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// #else
// #define HANDLE_ERROR(err) err
// #endif