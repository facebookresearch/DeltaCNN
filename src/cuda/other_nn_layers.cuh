// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdint.h>
#include <cuda_fp16.h>
#include "common.cuh"

void init_d_metrics_other_nn_layers();

template<typename scalar_t>
void activate_truncate(scalar_t* delta, scalar_t* prev_input, scalar_t* truncated, uint32_t *mask, float threshold, Dimensions dim, int activation, int truncation_mode);

template<typename scalar_t>
void activate_truncate_hp(scalar_t* delta, scalar_t* prev_input, scalar_t* truncated, uint32_t *mask, float threshold, Dimensions dim, int activation, int truncation_mode);

template<typename scalar_t>
void prepare_diff_mask(scalar_t* input, scalar_t* prev_input, scalar_t* delta, uint32_t *mask, float threshold, Dimensions dim);

template<typename scalar_t>
void prepare_diff_mask_hp(scalar_t* input, scalar_t* prev_input, scalar_t* delta, uint32_t *mask, float threshold, Dimensions dim);

template<typename scalar_t>
void sparse_add_tensors(scalar_t* a, scalar_t* b, scalar_t* prev_out, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, scalar_t weight_a, scalar_t weight_b, Dimensions dim, int activation, bool dense_out);

template<typename scalar_t>
void sparse_add_tensors_hp(scalar_t* a, scalar_t* b, scalar_t* prev_out, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, float weight_a, float weight_b, Dimensions dim, int activation, bool dense_out);

template<typename scalar_t>
void sparse_add_to_dense_tensor_sp(scalar_t* a, scalar_t* b, uint32_t *mask_a, Dimensions dim, int activation);

template<typename scalar_t>
void sparse_add_to_dense_tensor_hp(scalar_t* a, scalar_t* b, uint32_t *mask_a, Dimensions dim, int activation);

template<typename scalar_t>
void sparse_upsample(scalar_t* in, scalar_t* out, uint32_t *mask_in, uint32_t *mask_out, Dimensions dim, int scale);

template<typename scalar_t>
void sparse_upsample_hp(scalar_t* in, scalar_t* out, uint32_t *mask_in, uint32_t *mask_out, Dimensions dim, int scale);

template<typename scalar_t>
void sparse_concatenate(scalar_t* a, scalar_t* b, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, Dimensions dim);

template<typename scalar_t>
void sparse_concatenate_hp(scalar_t* a, scalar_t* b, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, Dimensions dim);

template<typename scalar_t>
void sparse_pool(scalar_t* input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config, int pooling_mode);

template<typename scalar_t>
void sparse_pool_hp(scalar_t* input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config, int pooling_mode);

template<typename scalar_t>
void sparse_mul_add(scalar_t* in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim);

template<typename scalar_t>
void sparse_mul_add_hp(scalar_t* in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim);