// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "common.cuh"

__device__ DCMetrics *d_metrics;

__global__ void check_d_metrics_ptr() {
    if (threadIdx.x == 0) {
        printf("common.cu &d_metrics=%p\n", d_metrics);
        printf("common.cu &d_metrics.vals_read_dense = %p\n", &d_metrics->n_vals_read_dense);
        printf("common.cu &d_metrics.vals_written_dense = %p\n", &d_metrics->n_vals_written_dense);
        d_metrics->n_active_flops += 1;
        printf("common.cu d_metrics.n_active_flops=%i\n", d_metrics->n_active_flops);
    }
}


bool init_performance_metrics() {
#ifdef ENABLE_METRICS
    HANDLE_ERROR(cudaMalloc(&d_metrics_ptr_copy, sizeof(DCMetrics)));
    HANDLE_ERROR(cudaMemset(d_metrics_ptr_copy, 0, sizeof(DCMetrics)));
    copy_performance_metrics_to_gpu(d_metrics);
    return true;
#else
    return false;
#endif
}

void copy_performance_metrics_to_gpu(DCMetrics*& d) {
#ifdef ENABLE_METRICS
    HANDLE_ERROR(cudaMemcpyToSymbol(d, &d_metrics_ptr_copy, sizeof(DCMetrics*)));
#endif
}

void reset_performance_metrics() {
#ifdef ENABLE_METRICS
    HANDLE_ERROR(cudaMemset(d_metrics_ptr_copy, 0, sizeof(DCMetrics)));
#endif
}

std::vector<torch::Tensor> retrieve_metrics() {
#ifdef ENABLE_METRICS    
    DCMetrics h_d_metric;
    HANDLE_ERROR(cudaMemcpy(&h_d_metric, d_metrics_ptr_copy, sizeof(DCMetrics), cudaMemcpyDeviceToHost));
    
    torch::Tensor tiles = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor inputs = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor mode = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor flops = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor memtransfer = torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor histrogram = torch::zeros({DCMetrics::histogram_samples}, torch::TensorOptions().dtype(torch::kInt64));

    tiles.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_active_tiles);
    tiles.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_tiles);
    inputs.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_active_inputs);
    inputs.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_inputs);
    mode.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_tiles_sparse_mode);
    mode.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_tiles_dense_mode);
    flops.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_active_flops);
    flops.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_theoretical_flops);
    flops.data_ptr<int64_t>()[2] = int64_t(h_d_metric.n_dense_flops);
    memtransfer.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_vals_read);
    memtransfer.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_vals_read_dense);
    memtransfer.data_ptr<int64_t>()[2] = int64_t(h_d_metric.n_vals_written);
    memtransfer.data_ptr<int64_t>()[3] = int64_t(h_d_metric.n_vals_written_dense);

    int64_t *histogram_ptr = histrogram.data_ptr<int64_t>();
    for (int i = 0; i < DCMetrics::histogram_samples; i++) {
        histogram_ptr[i] = int64_t(h_d_metric.active_input_histogram[i]);
    }
    return {tiles, inputs, mode, flops, memtransfer, histrogram};
#endif
    return {};
}