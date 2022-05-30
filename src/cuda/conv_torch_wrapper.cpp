// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include "conv_kernel.cuh"
#include "deconv_kernel.cuh"
#include "other_nn_layers.cuh"
#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include "common.cuh"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
// TODO add an actual check if the layer is channels last
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(!(x.is_contiguous()), #x " must be channels last")
#define CHECK_CONTIGUOUS(x) {}
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM((x).device().index() == (y).device().index(), #x " and " #y " must be in same CUDA device")

bool deltacnn_init_performance_metrics() {
    static bool initialized = false;
    if (initialized) {
        return true;
    }
    bool success = init_performance_metrics();
    if (!success)
        return success;
    initialized = true;
    init_d_metrics_conv_kernels();
    init_d_metrics_deconv_kernels();
    init_d_metrics_other_nn_layers();
    return success;    
}

ConvConfig params_to_conv_config(std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, int groups, int padding_mode, bool sub_tile_sparsity=true) {
    ConvConfig config;
    config.stride[0] = (uint16_t) stride[0];
    config.stride[1] = (uint16_t) stride[1];
    if (padding.size() == 4) {
        config.padding[0] = (uint16_t) padding[0];
        config.padding[1] = (uint16_t) padding[2];
        config.padding[2] = (uint16_t) padding[1];
        config.padding[3] = (uint16_t) padding[3];
    } else {
        config.padding[0] = (uint16_t) padding[0];
        config.padding[1] = (uint16_t) padding[1];
        config.padding[2] = (uint16_t) padding[0];
        config.padding[3] = (uint16_t) padding[1];
    }
    config.dilation[0] = (uint16_t) dilation[0];
    config.dilation[1] = (uint16_t) dilation[1];
    config.groups = groups;
    config.padding_mode = (PaddingMode) padding_mode;
    config.sub_tile_sparsity = sub_tile_sparsity;
    return config;
}

void sparse_conv_bias_wrapper_masked(
    torch::Tensor input,
    torch::Tensor filter,
    at::optional<torch::Tensor> bias,
    torch::Tensor out,
    at::optional<torch::Tensor> mask,
    at::optional<torch::Tensor> out_mask,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    int groups,
    int padding_mode,
    bool sub_tile_sparsity=true
    )
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    CHECK_INPUT(out);
    CHECK_DEVICE(input, filter);
    CHECK_DEVICE(input, out);

    uint32_t *out_mask_ptr = nullptr;
    uint32_t *mask_ptr = nullptr;
    void *bias_ptr = nullptr;

    if (bias) {
        CHECK_INPUT((*bias));
        CHECK_DEVICE(input, (*bias));
        if (input.dtype() == torch::kFloat32) {
            bias_ptr = (void*) (*bias).data_ptr<float>();
        } else {
            bias_ptr = (void*) (*bias).data_ptr<at::Half>();
        }
    }
    if (mask) {
        CHECK_INPUT((*mask));
        CHECK_DEVICE(input, (*mask));
        mask_ptr = (uint32_t*) (*mask).data_ptr<int32_t>();
    }
    if (out_mask) {
        CHECK_INPUT((*out_mask));
        CHECK_DEVICE(input, (*out_mask));
        out_mask_ptr = (uint32_t*) (*out_mask).data_ptr<int32_t>();
    }

    Dimensions dim;
    dim.batch_size = input.size(0);
    dim.in.c = input.size(1);
    dim.in.h = input.size(2);
    dim.in.w = input.size(3);
    dim.out.c = out.size(1);
    dim.out.h = out.size(2);
    dim.out.w = out.size(3);

    ConvConfig config = params_to_conv_config(stride, padding, dilation, groups, padding_mode, sub_tile_sparsity);
    config.kernel_size[0] = (uint8_t) filter.size(1);
    config.kernel_size[1] = (uint8_t) filter.size(2);
    config.set_sparse_zero = out_mask_ptr == nullptr;


    if (input.dtype() == torch::kFloat32) {
        deltacnn(input.data_ptr<float>(), out.data_ptr<float>(), filter.data_ptr<float>(), (float*) bias_ptr, (uint32_t*) mask_ptr, out_mask_ptr, dim, config);
    } else if (input.dtype() == torch::kFloat16) {
        deltacnn_hp((half*)input.data_ptr<at::Half>(), (half*)out.data_ptr<at::Half>(), (half*)filter.data_ptr<at::Half>(), (half*) bias_ptr, mask_ptr, out_mask_ptr, dim, config);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}

void sparse_deconv_bias_wrapper_masked(
    torch::Tensor input,
    torch::Tensor filter,
    at::optional<torch::Tensor> bias,
    torch::Tensor out,
    at::optional<torch::Tensor> mask,
    at::optional<torch::Tensor> out_mask,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    int groups,
    int padding_mode,
    bool sub_tile_sparsity=true
    )
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(input);
    CHECK_INPUT(filter);
    CHECK_INPUT(out);
    CHECK_DEVICE(input, filter);
    CHECK_DEVICE(input, out);

    uint32_t *out_mask_ptr = nullptr;
    uint32_t *mask_ptr = nullptr;
    void *bias_ptr = nullptr;

    if (bias) {
        CHECK_INPUT((*bias));
        CHECK_DEVICE(input, (*bias));
        if (input.dtype() == torch::kFloat32) {
            bias_ptr = (void*) (*bias).data_ptr<float>();
        } else {
            bias_ptr = (void*) (*bias).data_ptr<at::Half>();
        }
    }
    if (mask) {
        CHECK_INPUT((*mask));
        CHECK_DEVICE(input, (*mask));
        mask_ptr = (uint32_t*) (*mask).data_ptr<int32_t>();
    }
    if (out_mask) {
        CHECK_INPUT((*out_mask));
        CHECK_DEVICE(input, (*out_mask));
        out_mask_ptr = (uint32_t*) (*out_mask).data_ptr<int32_t>();
    }

    Dimensions dim;
    dim.batch_size = input.size(0);
    dim.in.c = input.size(1);
    dim.in.h = input.size(2);
    dim.in.w = input.size(3);
    dim.out.c = out.size(1);
    dim.out.h = out.size(2);
    dim.out.w = out.size(3);

    ConvConfig config = params_to_conv_config(stride, padding, dilation, groups, padding_mode, sub_tile_sparsity);
    config.kernel_size[0] = (uint8_t) filter.size(1);
    config.kernel_size[1] = (uint8_t) filter.size(2);
    config.set_sparse_zero = out_mask_ptr == nullptr;


    if (input.dtype() == torch::kFloat32) {
        delta_deconv(input.data_ptr<float>(), out.data_ptr<float>(), filter.data_ptr<float>(), (float*) bias_ptr, (uint32_t*) mask_ptr, out_mask_ptr, dim, config);
    } 
    else if (input.dtype() == torch::kFloat16) {
        delta_deconv_hp((half*)input.data_ptr<at::Half>(), (half*)out.data_ptr<at::Half>(), (half*)filter.data_ptr<at::Half>(), (half*) bias_ptr, mask_ptr, out_mask_ptr, dim, config);
    } 
    else {
        printf("unsupported datatype\n");
        return;
    }
}


void sparse_pooling_wrapper_masked(
    torch::Tensor input,
    at::optional<torch::Tensor> prev_in,
    torch::Tensor out,
    at::optional<torch::Tensor> mask,
    at::optional<torch::Tensor> out_mask,
    std::vector<int> kernel_size,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    int padding_mode,
    int pooling_mode,
    bool sub_tile_sparsity=true
    )
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(input);
    CHECK_INPUT(out);
    CHECK_DEVICE(input, out);

    void *prev_in_ptr = nullptr;
    uint32_t *out_mask_ptr = nullptr;
    uint32_t *mask_ptr = nullptr;

    if (prev_in) {
        CHECK_INPUT((*prev_in));
        CHECK_DEVICE(input, (*prev_in));
    }
    if (mask) {
        CHECK_INPUT((*mask));
        CHECK_DEVICE(input, (*mask));
        mask_ptr = (uint32_t*) (*mask).data_ptr<int32_t>();
    }
    if (out_mask) {
        CHECK_INPUT((*out_mask));
        CHECK_DEVICE(input, (*out_mask));
        out_mask_ptr = (uint32_t*) (*out_mask).data_ptr<int32_t>();
    }

    void *out_data_ptr_hp = nullptr;
    if (out.dtype() == torch::kFloat32) {
        out_data_ptr_hp = (void*) out.data_ptr<float>();
        if (prev_in) {
            prev_in_ptr = (void*) (*prev_in).data_ptr<float>();
        }
    } else if (out.dtype() == torch::kFloat16) {
        out_data_ptr_hp = (void*) out.data_ptr<at::Half>();
        if (prev_in) {
            prev_in_ptr = (void*) (*prev_in).data_ptr<at::Half>();
        }
    }

    ConvConfig config = params_to_conv_config(stride, padding, dilation, 1, padding_mode, sub_tile_sparsity);
    config.kernel_size[0] = (uint16_t) kernel_size[0];
    config.kernel_size[1] = (uint16_t) kernel_size[1];
    config.set_sparse_zero = out_mask_ptr == nullptr;

    Dimensions dim;
    dim.batch_size = input.size(0);
    dim.in.c = input.size(1);
    dim.in.h = input.size(2);
    dim.in.w = input.size(3);
    dim.out.c = out.size(1);
    dim.out.h = out.size(2);
    dim.out.w = out.size(3);


    // printf("out ptr address=%p\n", (void*) out_data_ptr_hp);
    if (input.dtype() == torch::kFloat32) {
        sparse_pool(input.data_ptr<float>(), (float*) prev_in_ptr, out.data_ptr<float>(), (uint32_t*) mask_ptr, out_mask_ptr, dim, config, pooling_mode);
    } else if (input.dtype() == torch::kFloat16) {
        sparse_pool_hp((half*) input.data_ptr<at::Half>(), (half*) prev_in_ptr, (half*) out_data_ptr_hp, (uint32_t*) mask_ptr, out_mask_ptr, dim, config, pooling_mode);
    } else {
        printf("unsupported datatype\n");
        return;
    }
    // printf("out ptr address=%p\n", (void*) out_data_ptr_hp);
}

void deltacnn_activate_truncate(
    torch::Tensor input,
    torch::Tensor prev_input,
    at::optional<torch::Tensor> truncated,
    torch::Tensor mask,
    float threshold,
    int activation,
    int truncation_mode
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(input);
    CHECK_INPUT(prev_input);
    float *truncated_ptr = nullptr;
    half *truncated_ptr_hp = nullptr;
    if (truncated) {
        CHECK_INPUT(*truncated);
        CHECK_DEVICE(input, *truncated);
        if (input.dtype() == torch::kFloat32)
            truncated_ptr = (float*) (*truncated).data_ptr<float>();
        if (input.dtype() == torch::kFloat16)
            truncated_ptr_hp = (half*) (*truncated).data_ptr<at::Half>();
    }
    CHECK_DEVICE(input, prev_input);
    CHECK_INPUT(mask);
    CHECK_DEVICE(input, mask);
    Dimensions dim;
    dim.batch_size = input.size(0);
    dim.in.h = input.size(2);
    dim.in.w = input.size(3);
    dim.in.c = input.size(1);
    dim.out.h = input.size(2);
    dim.out.w = input.size(3);
    dim.out.c = input.size(1);

    

    if (input.dtype() == torch::kFloat32) {
        activate_truncate<float>(input.data_ptr<float>(), prev_input.data_ptr<float>(), truncated_ptr, (uint32_t*) mask.data_ptr<int32_t>(), threshold, dim, activation, truncation_mode);
    } else if (input.dtype() == torch::kFloat16) {
        activate_truncate_hp<half>((half*) input.data_ptr<at::Half>(), (half*) prev_input.data_ptr<at::Half>(), truncated_ptr_hp, (uint32_t*) mask.data_ptr<int32_t>(), threshold, dim, activation, truncation_mode);
    }
    else {
        printf("unsupported datatype\n");
        return;
    }
}


void deltacnn_prepare_diff_mask_wrapper(
    torch::Tensor input,
    torch::Tensor prev_input,
    torch::Tensor delta,
    torch::Tensor mask,
    float threshold
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(input);
    CHECK_INPUT(prev_input);
    CHECK_INPUT(delta);
    CHECK_INPUT(mask);
    CHECK_DEVICE(input, prev_input);
    CHECK_DEVICE(input, delta);
    CHECK_DEVICE(input, mask);

    Dimensions dim;
    dim.batch_size = input.size(0);
    dim.in.c = input.size(1);
    dim.in.h = input.size(2);
    dim.in.w = input.size(3);
    dim.out.c = input.size(1);
    dim.out.h = input.size(2);
    dim.out.w = input.size(3);

    if (input.dtype() == torch::kFloat32) {
        prepare_diff_mask<float>(input.data_ptr<float>(), prev_input.data_ptr<float>(), delta.data_ptr<float>(), (uint32_t*) mask.data_ptr<int32_t>(), threshold, dim);
    } else if (input.dtype() == torch::kFloat16) {
        prepare_diff_mask_hp<half>((half*) input.data_ptr<at::Half>(), (half*) prev_input.data_ptr<at::Half>(), (half*) delta.data_ptr<at::Half>(), (uint32_t*) mask.data_ptr<int32_t>(), threshold, dim);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}


void sparse_add_tensors_wrapper(
    torch::Tensor a,
    torch::Tensor b,
    at::optional<torch::Tensor> prev_out,
    torch::Tensor out,
    torch::Tensor mask_a,
    torch::Tensor mask_b,
    torch::Tensor mask_out,
    float weight_a,
    float weight_b,
    int activation,
    bool dense_out
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(out);
    CHECK_INPUT(mask_a);
    CHECK_INPUT(mask_b);
    CHECK_INPUT(mask_out);
    CHECK_DEVICE(a, b);
    CHECK_DEVICE(a, out);
    CHECK_DEVICE(a, mask_a);
    CHECK_DEVICE(a, mask_b);
    CHECK_DEVICE(a, mask_out);

    Dimensions dim;
    dim.batch_size = a.size(0);
    dim.in.c = a.size(1);
    dim.in.h = a.size(2);
    dim.in.w = a.size(3);
    dim.out.c = a.size(1);
    dim.out.h = a.size(2);
    dim.out.w = a.size(3);

    void *prev_out_ptr = nullptr;
    if (prev_out) {
        CHECK_INPUT((*prev_out));
        CHECK_DEVICE(a, (*prev_out));

        if ((*prev_out).dtype() == torch::kFloat32) {
            prev_out_ptr = (void*) (*prev_out).data_ptr<float>();
        }
    }

    if (a.dtype() == torch::kFloat32) {
        sparse_add_tensors<float>(a.data_ptr<float>(), b.data_ptr<float>(), (float*) prev_out_ptr, out.data_ptr<float>(), (uint32_t*) mask_a.data_ptr<int32_t>(), (uint32_t*) mask_b.data_ptr<int32_t>(), (uint32_t*) mask_out.data_ptr<int32_t>(), weight_a, weight_b, dim, activation, dense_out);
    } else if (a.dtype() == torch::kFloat16) {
        sparse_add_tensors_hp<half>((half*) a.data_ptr<at::Half>(), (half*) b.data_ptr<at::Half>(), (half*) prev_out_ptr, (half*) out.data_ptr<at::Half>(), (uint32_t*) mask_a.data_ptr<int32_t>(), (uint32_t*) mask_b.data_ptr<int32_t>(), (uint32_t*) mask_out.data_ptr<int32_t>(), weight_a, weight_b, dim, activation, dense_out);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}


void sparse_add_to_dense_tensor_wrapper(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor mask_a,
    int activation
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(mask_a);
    CHECK_DEVICE(a, b);
    CHECK_DEVICE(a, mask_a);

    Dimensions dim;
    dim.batch_size = a.size(0);
    dim.in.c = a.size(1);
    dim.in.h = a.size(2);
    dim.in.w = a.size(3);
    dim.out.c = a.size(1);
    dim.out.h = a.size(2);
    dim.out.w = a.size(3);

    if (a.dtype() == torch::kFloat32) {
        sparse_add_to_dense_tensor_sp<float>(a.data_ptr<float>(), b.data_ptr<float>(), (uint32_t*) mask_a.data_ptr<int32_t>(), dim, activation);
    } else if (a.dtype() == torch::kFloat16) {
        sparse_add_to_dense_tensor_hp<half>((half*) a.data_ptr<at::Half>(), (half*) b.data_ptr<at::Half>(),(uint32_t*) mask_a.data_ptr<int32_t>(), dim, activation);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}


void sparse_upsample_wrapper(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor mask_in,
    torch::Tensor mask_out,
    int scale
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(input);
    CHECK_INPUT(out);
    CHECK_INPUT(mask_in);
    CHECK_INPUT(mask_out);
    CHECK_DEVICE(input, out);
    CHECK_DEVICE(input, mask_in);
    CHECK_DEVICE(input, mask_out);

    Dimensions dim;
    dim.batch_size = input.size(0);
    dim.in.c = input.size(1);
    dim.in.h = input.size(2);
    dim.in.w = input.size(3);
    dim.out.c = out.size(1);
    dim.out.h = out.size(2);
    dim.out.w = out.size(3);

    if (input.dtype() == torch::kFloat32) {
        sparse_upsample<float>(input.data_ptr<float>(), out.data_ptr<float>(), (uint32_t*) mask_in.data_ptr<int32_t>(), (uint32_t*) mask_out.data_ptr<int32_t>(), dim, scale);
    } else if (input.dtype() == torch::kFloat16) {
        sparse_upsample_hp<half>((half*) input.data_ptr<at::Half>(), (half*) out.data_ptr<at::Half>(), (uint32_t*) mask_in.data_ptr<int32_t>(), (uint32_t*) mask_out.data_ptr<int32_t>(), dim, scale);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}


void sparse_concatenate_wrapper(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor out,
    torch::Tensor mask_a,
    torch::Tensor mask_b,
    torch::Tensor mask_out
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(out);
    CHECK_INPUT(mask_a);
    CHECK_INPUT(mask_b);
    CHECK_INPUT(mask_out);
    CHECK_DEVICE(a, b);
    CHECK_DEVICE(a, out);
    CHECK_DEVICE(a, mask_a);
    CHECK_DEVICE(a, mask_b);
    CHECK_DEVICE(a, mask_out);

    Dimensions dim;
    dim.batch_size = a.size(0);
    dim.in.c = a.size(1);
    dim.in.h = a.size(2);
    dim.in.w = a.size(3);
    dim.out.c = out.size(1);
    dim.out.h = out.size(2);
    dim.out.w = out.size(3);

    if (a.dtype() == torch::kFloat32) {
        sparse_concatenate<float>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), (uint32_t*) mask_a.data_ptr<int32_t>(), (uint32_t*) mask_b.data_ptr<int32_t>(), (uint32_t*) mask_out.data_ptr<int32_t>(), dim);
    } else if (a.dtype() == torch::kFloat16) {
        sparse_concatenate_hp<half>((half*) a.data_ptr<at::Half>(), (half*) b.data_ptr<at::Half>(), (half*) out.data_ptr<at::Half>(), (uint32_t*) mask_a.data_ptr<int32_t>(), (uint32_t*) mask_b.data_ptr<int32_t>(), (uint32_t*) mask_out.data_ptr<int32_t>(), dim);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}


void sparse_mul_add_wrapper(
    torch::Tensor in,
    torch::Tensor out,
    at::optional<torch::Tensor> mask_in,
    at::optional<torch::Tensor> mask_out,
    torch::Tensor scale,
    at::optional<torch::Tensor> bias
    ) 
{
#ifdef ENABLE_METRICS
    deltacnn_init_performance_metrics();
#endif

    CHECK_INPUT(in);
    CHECK_INPUT(out);
    CHECK_INPUT(scale);
    CHECK_DEVICE(in, out);
    CHECK_DEVICE(in, scale);

    void *bias_ptr = nullptr;
    if (bias) {
        CHECK_INPUT(*bias);
        CHECK_DEVICE(in, *bias);
        if (bias->dtype() == torch::kFloat32) {
            bias_ptr = (void*) bias->data_ptr<float>();
        } else if (bias->dtype() == torch::kFloat16) {
            bias_ptr = (void*) bias->data_ptr<at::Half>();
        }
    }

    uint32_t *mask_in_ptr = nullptr;
    if (mask_in) {
        CHECK_INPUT(*mask_in);
        CHECK_DEVICE(in, *mask_in);
        mask_in_ptr = (uint32_t*) mask_in->data_ptr<int32_t>();
    }

    uint32_t *mask_out_ptr = nullptr;
    if (mask_out) {
        CHECK_INPUT(*mask_out);
        CHECK_DEVICE(in, *mask_out);
        mask_out_ptr = (uint32_t*) mask_out->data_ptr<int32_t>();
    }

    Dimensions dim;
    dim.batch_size = in.size(0);
    dim.in.c = in.size(1);
    dim.in.h = in.size(2);
    dim.in.w = in.size(3);
    dim.out.c = in.size(1);
    dim.out.h = in.size(2);
    dim.out.w = in.size(3);

    if (in.dtype() == torch::kFloat32) {
        sparse_mul_add<float>(in.data_ptr<float>(), mask_in_ptr, out.data_ptr<float>(), mask_out_ptr, scale.data_ptr<float>(), (float*) bias_ptr, dim);
    } else if (in.dtype() == torch::kFloat16) {
        sparse_mul_add_hp<half>((half*) in.data_ptr<at::Half>(), mask_in_ptr, (half*) out.data_ptr<at::Half>(), mask_out_ptr, (half*) scale.data_ptr<at::Half>(), (half*) bias_ptr, dim);
    } else {
        printf("unsupported datatype\n");
        return;
    }
}

void deltacnn_reset_performance_metrics() {
    reset_performance_metrics();
}

std::vector<torch::Tensor> deltacnn_retrieve_metrics() {
    return retrieve_metrics();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_conv_bias_wrapper_masked", &sparse_conv_bias_wrapper_masked, "Sparse Convolution with Bias PyTorch wrapper with per pixel mask (CUDA)");
    m.def("sparse_deconv_bias_wrapper_masked", &sparse_deconv_bias_wrapper_masked, "Sparse Transposed Convolution with Bias PyTorch wrapper with per pixel mask (CUDA)");
    m.def("sparse_pooling_wrapper_masked", &sparse_pooling_wrapper_masked, "Sparse Pooling (CUDA)");
    m.def("deltacnn_activate_truncate", &deltacnn_activate_truncate, "Aggregate previous and current inputs (CUDA)");
    m.def("deltacnn_prepare_diff_mask_wrapper", &deltacnn_prepare_diff_mask_wrapper, "Calculate diff mask and update previous input (CUDA)");
    m.def("sparse_add_tensors_wrapper", &sparse_add_tensors_wrapper, "Add 2 sparse tensors and create a union of their mask (CUDA)");
    m.def("sparse_add_to_dense_tensor_wrapper", &sparse_add_to_dense_tensor_wrapper, "Add sparse tensor updates to dense tensor (CUDA)");
    m.def("sparse_mul_add_wrapper", &sparse_mul_add_wrapper, "Apply scale and bias to sparse tensor (CUDA)");
    m.def("sparse_upsample_wrapper", &sparse_upsample_wrapper, "Upsample an image with integer scales (CUDA)");
    m.def("sparse_concatenate_wrapper", &sparse_concatenate_wrapper, "Concatenated Tensors A and B along the channels (dim=1) (CUDA)");
    m.def("deltacnn_init_performance_metrics", &deltacnn_init_performance_metrics, "Init DeltaCNN performance metrics. Returns false if metrics are disabled by compile flags");
    m.def("deltacnn_reset_performance_metrics", &deltacnn_reset_performance_metrics, "Reset DeltaCNN performance metrics");
    m.def("deltacnn_retrieve_metrics", &deltacnn_retrieve_metrics, "Get performance metrics. In this order: tiles, inputs, mode, flops, memtransfer, histogram");
}