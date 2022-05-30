// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdint.h>
#include <cuda_fp16.h>
#include "common.cuh"
#include <vector>


template<typename scalar_t>
void deltacnn(scalar_t *input, scalar_t *output, scalar_t *filter, scalar_t *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config);

void deltacnn_hp(half *input, half *output, half *filter, half *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config);

void init_d_metrics_conv_kernels();