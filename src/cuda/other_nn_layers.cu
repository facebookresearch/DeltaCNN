// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "other_nn_layers.cuh"
#include "Utility.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ DCMetrics *d_metrics;

#define divup(a, b) (((a) + (b) - 1) / (b))

void init_d_metrics_other_nn_layers() {
#ifdef ENABLE_METRICS
    copy_performance_metrics_to_gpu(d_metrics);
#endif
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t relu(scalar_t x) {
    return x < 0.0f ? 0.0f : x; 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t relu6(scalar_t x) {
    return x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x); 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t leaky_relu(scalar_t x) {
    return x < 0.0f ? x * 0.1f : x; 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t x) {
    return 1.0f / (1.0f + expf(-x)); 
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t swish(scalar_t x) {
    return x / (1.0f + expf(-x)); 
}

__device__ __forceinline__ half hrelu(half x) {
    half zero = __float2half(0.0f);
    return __hlt(x, zero) ? zero : x;
}

__device__ __forceinline__ half hrelu6(half x) {
    half zero = __float2half(0.0f), six = __float2half(6.0f);
    return __hlt(x, zero) ? zero : (__hgt(x, six) ? six : x);
}

__device__ __forceinline__ half hleaky_relu(half x) {
    half zero = __float2half(0.0f);
    return __hlt(x, zero) ? __hmul(x, __float2half(0.1)) : x;
}

__device__ __forceinline__ half hsigmoid(half x) {
    return hrcp(__hadd(__float2half(1.0f), hexp(__hneg(x)))); 
}

__device__ __forceinline__ half hswish(half x) {
    return __hdiv(x, __hadd(__float2half(1.0f), hexp(__hneg(x)))); 
}

__device__ __forceinline__ half2 hrelu_2(half2 x) {
    half x_low = __low2half(x), x_high = __high2half(x);
    half zero = __float2half(0.0f);
    return __halves2half2(
        __hlt(x_low, zero) ? zero : x_low,
        __hlt(x_high, zero) ? zero : x_high
    );
}

__device__ __forceinline__ half2 hrelu6_2(half2 x) {
    half x_low = __low2half(x), x_high = __high2half(x);
    half zero = __float2half(0.0f), six = __float2half(6.0f);
    return __halves2half2(
        __hlt(x_low, zero) ? zero : 
            (__hgt(x_low, six) ? six : x_low),
        __hlt(x_high, zero) ? zero : 
            (__hgt(x_high, six) ? six : x_high)
    );
}

__device__ __forceinline__ half2 hleaky_relu_2(half2 x) {
    half x_low = __low2half(x), x_high = __high2half(x);
    half zero = __float2half(0.0f);
    return __halves2half2(
        __hlt(x_low, zero) ? __hmul(x_low, __float2half(0.1)) : x_low,
        __hlt(x_high, zero) ? __hmul(x_high, __float2half(0.1)) : x_high
    );
}

__device__ __forceinline__ half2 hsigmoid_2(half2 x) {
    return h2rcp(__hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(x)))); 
}

__device__ __forceinline__ half2 hswish_2(half2 x) {
    return __h2div(x, __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(x)))); 
}

template <int mode, typename scalar_t>
__device__ __forceinline__ scalar_t activation_selector(scalar_t x) {
    if (mode == 1) {
        return relu(x);
    } else if (mode == 2) {
        return relu6(x);
    } else if (mode == 3) {
        return leaky_relu(x);
    } else if (mode == 4) {
        return sigmoid(x);
    } else if (mode == 5) {
        return swish(x);
    } else {
        return x;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t activation_selector(scalar_t x, int mode) {
    if (mode == 1) {
        return relu(x);
    } else if (mode == 2) {
        return relu6(x);
    } else if (mode == 3) {
        return leaky_relu(x);
    } else if (mode == 4) {
        return sigmoid(x);
    } else if (mode == 5) {
        return swish(x);
    } else {
        return x;
    }
}

template<>
__device__ __forceinline__ float4 activation_selector<0, float4>(float4 x) {
    return x;
}

template<>
__device__ __forceinline__ float4 activation_selector<1, float4>(float4 x) {
    float4 result;
    result.x = relu(x.x);
    result.y = relu(x.y);
    result.z = relu(x.z);
    result.w = relu(x.w);
    
    return result;
}

template<>
__device__ __forceinline__ float4 activation_selector<2, float4>(float4 x) {
    float4 result;
    result.x = relu6(x.x);
    result.y = relu6(x.y);
    result.z = relu6(x.z);
    result.w = relu6(x.w);
    
    return result;
}

template<>
__device__ __forceinline__ float4 activation_selector<3, float4>(float4 x) {
    float4 result;
    result.x = leaky_relu(x.x);
    result.y = leaky_relu(x.y);
    result.z = leaky_relu(x.z);
    result.w = leaky_relu(x.w);
    
    return result;
}

template<>
__device__ __forceinline__ float4 activation_selector<4, float4>(float4 x) {
    float4 result;
    result.x = sigmoid(x.x);
    result.y = sigmoid(x.y);
    result.z = sigmoid(x.z);
    result.w = sigmoid(x.w);
    
    return result;
}

template<>
__device__ __forceinline__ float4 activation_selector<5, float4>(float4 x) {
    float4 result;
    result.x = swish(x.x);
    result.y = swish(x.y);
    result.z = swish(x.z);
    result.w = swish(x.w);
    
    return result;
}

// template<int mode>
// __device__ __forceinline__ float2 activation_selector<mode, float2>(float2 x) {
//     float2 result;

//     if (mode == 1) {
//         result.x =  relu(x.x);
//         result.y =  relu(x.y);
//     } else if (mode == 2) {
//         result.x =  relu6(x.x);
//         result.y =  relu6(x.y);
//     } else if (mode == 3) {
//         result.x =  leaky_relu(x.x);
//         result.y =  leaky_relu(x.y);
//     } else if (mode == 4) {
//         result.x =  sigmoid(x.x);
//         result.y =  sigmoid(x.y);
//     } else if (mode == 5) {
//         result.x =  swish(x.x);
//         result.y =  swish(x.y);
//     } else {
//         return x;
//     }
    
//     return result;
// }

template<>
__device__ __forceinline__ float2 activation_selector<0, float2>(float2 x) {
    return x;
}

template<>
__device__ __forceinline__ float2 activation_selector<1, float2>(float2 x) {
    float2 result;
    result.x = relu(x.x);
    result.y = relu(x.y);
    
    return result;
}

template<>
__device__ __forceinline__ float2 activation_selector<2, float2>(float2 x) {
    float2 result;
    result.x = relu6(x.x);
    result.y = relu6(x.y);

    return result;
}

template<>
__device__ __forceinline__ float2 activation_selector<3, float2>(float2 x) {
    float2 result;
    result.x = leaky_relu(x.x);
    result.y = leaky_relu(x.y);
    return result;
}

template<>
__device__ __forceinline__ float2 activation_selector<4, float2>(float2 x) {
    float2 result;
    result.x = sigmoid(x.x);
    result.y = sigmoid(x.y);
    
    return result;
}

template<>
__device__ __forceinline__ float2 activation_selector<5, float2>(float2 x) {
    float2 result;
    result.x = swish(x.x);
    result.y = swish(x.y);
    
    return result;
}

template<int mode>
__device__ __forceinline__ void activation_selector_float4(float *x, float *result) {
    if (mode == 1) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] = relu(x[i]);
        }
    } else if (mode == 2) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] = relu6(x[i]);
        }
    } else if (mode == 3) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] = leaky_relu(x[i]);
        }
    } else if (mode == 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] = sigmoid(x[i]);
        }
    } else if (mode == 5) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] = swish(x[i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] = x[i];
        }
    }
}

template<int mode>
__device__ __forceinline__ half hactivation_selector(half x) {
    if (mode == 1) {
        return hrelu(x);
    } else if (mode == 2) {
        return hrelu6(x);
    } else if (mode == 3) {
        return hleaky_relu(x);
    } else if (mode == 4) {
        return hsigmoid(x);
    } else if (mode == 5) {
        return hswish(x);
    } else {
        return x;
    }
}

__device__ __forceinline__ half hactivation_selector(half x, int mode) {
    if (mode == 1) {
        return hrelu(x);
    } else if (mode == 2) {
        return hrelu6(x);
    } else if (mode == 3) {
        return hleaky_relu(x);
    } else if (mode == 4) {
        return hsigmoid(x);
    } else if (mode == 5) {
        return hswish(x);
    } else {
        return x;
    }
}

template<int mode>
__device__ __forceinline__ half2 hactivation_selector2(half2 x) {
    if (mode == 1) {
        return hrelu_2(x);
    } else if (mode == 2) {
        return hrelu6_2(x);
    } else if (mode == 3) {
        return hleaky_relu_2(x);
    } else if (mode == 4) {
        return hsigmoid_2(x);
    } else if (mode == 5) {
        return hswish_2(x);
    } else {
        return x;
    }
}


__device__ __forceinline__ half2 hactivation_selector2(half2 x, int mode) {
    if (mode == 1) {
        return hrelu_2(x);
    } else if (mode == 2) {
        return hrelu6_2(x);
    } else if (mode == 3) {
        return hleaky_relu_2(x);
    } else if (mode == 4) {
        return hsigmoid_2(x);
    } else if (mode == 5) {
        return hswish_2(x);
    } else {
        return x;
    }
}


template <typename scalar_t = float, int activation>
__device__ void activate_no_truncation(scalar_t *delta_px, scalar_t *prev_input_px, Dimensions dim)
{
    const int lane_idx = threadIdx.x % WARP_SIZE;

    for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
        scalar_t delta = delta_px[c];
        scalar_t prev_in = prev_input_px[c];
        scalar_t sum = delta + prev_in;
        
        prev_input_px[c] = sum;
        if (activation > 0) {
            delta = activation_selector(sum, activation) - activation_selector(prev_in, activation);
            delta_px[c] = delta;
        }
    }
}


template <typename scalar_t = float, int activation>
__device__ bool activate_truncate_max(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, Dimensions dim, float threshold)
{
    bool pixel_active = false;
    int first_pixel_active = -999;
    const int lane_idx = threadIdx.x % WARP_SIZE;

    for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE) {
        int c = c_off + lane_idx;
        scalar_t delta, sum, trunc;

        const bool inside = c < dim.in.c;

        if (inside) {
            delta = delta_px[c];
            scalar_t prev_in = prev_input_px[c];
            trunc = truncated_px[c];
            trunc += delta;
            sum = trunc + prev_in;
            
            delta = activation_selector<activation>(sum) - activation_selector<activation>(prev_in);
        } else {
            delta = 0.0f;
        }

        if (!pixel_active) {
            pixel_active |= fabsf(delta) > threshold;
            pixel_active = __any_sync(FULL_MASK, pixel_active);
            if (pixel_active) {
                first_pixel_active = c;
            }
        }

        if (pixel_active) {
            if (inside) {
                prev_input_px[c] = sum;
                delta_px[c] = delta;
                truncated_px[c] = 0.0f;
            }
        } else {
            if (inside) {
                truncated_px[c] = trunc;
            }
        }
    }

    for (int c = lane_idx; c < first_pixel_active && c < dim.in.c; c += WARP_SIZE) {
        scalar_t prev_in = prev_input_px[c];
        scalar_t trunc = truncated_px[c];
        scalar_t sum = trunc + prev_in;
        
        if (activation > 0) {
            scalar_t delta;
            delta = activation_selector<activation>(sum) - activation_selector<activation>(prev_in);
            delta_px[c] = delta;
        }

        prev_input_px[c] = sum;
        truncated_px[c] = 0.0f;
    }
    
    if (lane_idx == 0 && !pixel_active) {
        *mask_px = 0;
    }

    return pixel_active;
}

template<typename T>
__device__ __forceinline__ T add_float_vec(T a, T b) {
}

template<>
__device__ __forceinline__ float4 add_float_vec<float4>(float4 a, float4 b) {
    float4 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    res.w = a.w + b.w;
    return res;
}

template<>
__device__ __forceinline__ float2 add_float_vec<float2>(float2 a, float2 b) {
    float2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

template<typename T>
__device__ __forceinline__ T sub_float_vec(T a, T b) {
}

template<>
__device__ __forceinline__ float4 sub_float_vec<float4>(float4 a, float4 b) {
    float4 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    res.z = a.z - b.z;
    res.w = a.w - b.w;
    return res;
}

template<>
__device__ __forceinline__ float2 sub_float_vec<float2>(float2 a, float2 b) {
    float2 res;
    res.x = a.x - b.x;
    res.y = a.y - b.y;
    return res;
}

template<typename T>
__device__ __forceinline__ void add_float_vec_inline(T& a, T b) {
    a += b;
}

template<>
__device__ __forceinline__ void add_float_vec_inline<float4>(float4& a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

template<>
__device__ __forceinline__ void add_float_vec_inline<float2>(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

template<typename T>
__device__ __forceinline__ bool any_larger_than_abs(T a, float t) {
}

template<>
__device__ __forceinline__ bool any_larger_than_abs<float4>(float4 a, float t) {
    return (fabsf(a.x) > t) || (fabsf(a.y) > t) || (fabsf(a.z) > t) || (fabsf(a.w) > t);
}

template<>
__device__ __forceinline__ bool any_larger_than_abs<float2>(float2 a, float t) {
    return (fabsf(a.x) > t) || (fabsf(a.y) > t);
}


__device__ __forceinline__ void add_float4(float *a, float *b, float *res) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        res[i] = a[i] + b[i];
    }
}

__device__ __forceinline__ void sub_float4(float *a, float *b, float *res) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        res[i] = a[i] - b[i];
    }
}

__device__ __forceinline__ void add_float4_inline(float *a, float *b) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] += b[i];
    }
}

template<typename T>
__device__ __forceinline__ T make_zeros() {
    return T(0);
}

template<>
__device__ __forceinline__ float4 make_zeros<float4>() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template<>
__device__ __forceinline__ float2 make_zeros<float2>() {
    return make_float2(0.0f, 0.0f);
}


template <typename scalar_t = float4, int activation, int floats_per_thread>
__device__ bool activate_truncate_max_vectorized(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, Dimensions dim, float threshold)
{
    bool pixel_active = false;
    int first_pixel_active = -999;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int iters = dim.in.c / floats_per_thread;

    for (int c_off = 0; c_off < iters; c_off += WARP_SIZE) {
        int c = c_off + lane_idx;
        scalar_t delta, sum, trunc;
        const bool inside = c < iters; 

        if (inside) {
            delta = delta_px[c];
            scalar_t prev_in = prev_input_px[c];
            trunc = truncated_px[c];

            add_float_vec_inline(trunc, delta);
            sum = add_float_vec(trunc, prev_in);
            
            delta = sub_float_vec(activation_selector<activation>(sum), activation_selector<activation>(prev_in));
        } else {
            delta = make_zeros<scalar_t>();
        }

        pixel_active |= any_larger_than_abs(delta, threshold);
        pixel_active = __any_sync(FULL_MASK, pixel_active);
        if (pixel_active) {
            first_pixel_active = c;
            if (inside) {
                prev_input_px[c] = sum;
                delta_px[c] = delta;
                truncated_px[c] = make_zeros<scalar_t>();
            }
            break;
        } else {
            if (inside) {
                truncated_px[c] = trunc;
            }
        }
    }

    if (first_pixel_active >= 0) {
        for (int c = first_pixel_active + WARP_SIZE; c < iters; c += WARP_SIZE) {
            scalar_t delta, sum, trunc;

            delta = delta_px[c];
            scalar_t prev_in = prev_input_px[c];
            trunc = truncated_px[c];

            add_float_vec_inline(trunc, delta);
            sum = add_float_vec(trunc, prev_in);
            
            delta = sub_float_vec(activation_selector<activation>(sum), activation_selector<activation>(prev_in));
            prev_input_px[c] = sum;
            delta_px[c] = delta;
            truncated_px[c] = make_zeros<scalar_t>();
        }

        for (int c = lane_idx; c < first_pixel_active && c < iters; c += WARP_SIZE) {
            scalar_t prev_in = prev_input_px[c];
            scalar_t trunc = truncated_px[c];
            scalar_t sum = add_float_vec(trunc, prev_in);
            
            if (activation > 0) {
                scalar_t delta;
                delta = sub_float_vec(activation_selector<activation>(sum), activation_selector<activation>(prev_in));
                delta_px[c] = delta;
            }

            prev_input_px[c] = sum;
            truncated_px[c] = make_zeros<scalar_t>();
        }
    }

    if (lane_idx == 0 && !pixel_active) {
        *mask_px = 0;
    }

    return pixel_active;
}


template <typename scalar_t = float, int activation>
__device__ bool activate_truncate_sum(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, Dimensions dim, float threshold, int truncation_mode)
{
    const int lane_idx = threadIdx.x % WARP_SIZE;
    bool pixel_active = false;

    scalar_t pixel_sum = scalar_t(0.0f);

    for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
        scalar_t delta = delta_px[c];
        scalar_t prev_in = prev_input_px[c];
        scalar_t trunc = truncated_px[c];
        trunc += delta;
        scalar_t sum = trunc + prev_in;
        
        delta = activation_selector<activation>(sum) - activation_selector<activation>(prev_in);
        pixel_sum += delta * delta;
        truncated_px[c] = trunc;
    }

    pixel_sum = Utils::warpReduceSum(pixel_sum);
    if (truncation_mode == 1) {
        // euclidian norm
        pixel_active = sqrt(pixel_sum) > threshold;
    } else if (truncation_mode == 2) {
        // rmse
        pixel_active = sqrt(pixel_sum / dim.in.c) > threshold;
    } else {
        pixel_active = true;
        if (threadIdx.x == 0) {
            printf("Truncation mode %i not implemented\n", truncation_mode);
        }
    }
    pixel_active = __shfl_sync(FULL_MASK, pixel_active, 0);

    if (pixel_active) {
        for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
            scalar_t prev_in = prev_input_px[c];
            scalar_t trunc = truncated_px[c];
            scalar_t sum = trunc + prev_in;
            
            if (activation > 0) {
                scalar_t delta;
                delta = activation_selector<activation>(sum) - activation_selector<activation>(prev_in);
                delta_px[c] = delta;
            }

            prev_input_px[c] = sum;
            truncated_px[c] = 0.0f;
        }
    } else {
        *mask_px = 0;
    }

    return pixel_active;
}

template<int PIXELS_PER_BLOCK, int BLOCK_SIZE>
__device__ __forceinline__ bool checkIfAnyActive(const int start_idx, const int end_idx, const int warp_idx, const uint32_t* mask) {
    if (PIXELS_PER_BLOCK <= 32) {
        int px_idx = start_idx + (threadIdx.x % WARP_SIZE);
        uint32_t active = px_idx < end_idx ? mask[px_idx] : 0;
        active = __ballot_sync(0xFFFFFFFF, active);
        return active != 0;
    } else {
        __shared__ int any_active;
        if (threadIdx.x == 0) {
            any_active = 0;
        }
        __syncthreads();

        int lane_idx = threadIdx.x % WARP_SIZE;
        for (int px_idx = warp_idx * WARP_SIZE + start_idx; px_idx < end_idx; px_idx += BLOCK_SIZE) {
            int px_idx_thread = px_idx + lane_idx;
            uint32_t active = px_idx_thread < end_idx ? mask[px_idx_thread] : 0;
            active = __ballot_sync(0xFFFFFFFF, active);
            if (active != 0) {
                if (lane_idx == 0)
                    any_active = 1;
                break;
            }
            if (any_active != 0) {
                break;
            }
        }
        __syncthreads();
        return any_active != 0;
    }
}

template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, int activation, int vectorization = 1>
__global__ void deltacnn_activate_truncate_kernel(scalar_t * __restrict__ delta, scalar_t * __restrict__ prev_input, scalar_t * __restrict__ truncated, uint32_t *mask, float threshold, Dimensions dim, int truncation_mode) {
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

    if (!checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask)) {
        return;
    }

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {  
#ifdef ENABLE_METRICS
        if (threadIdx.x % WARP_SIZE == 0) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.in.c)); 
        }
#endif
        if (mask[px_idx] == 0)
            continue;

        scalar_t *delta_px = &delta[px_idx * dim.in.c];
        scalar_t *prev_input_px = &prev_input[px_idx * dim.in.c];
        scalar_t *truncated_px = &truncated[px_idx * dim.in.c];
        uint32_t *mask_px = &mask[px_idx];

        [[maybe_unused]]bool active = true;

        if (truncation_mode < 0) {
            // no truncation
            activate_no_truncation<scalar_t, activation>(delta_px, prev_input_px, dim);
        }
        else if (truncation_mode == 0) {
            // max value > threshold
            if (vectorization == 4) {
                active = activate_truncate_max_vectorized<float4, activation, 4>(
                    reinterpret_cast<float4*>(delta_px), 
                    reinterpret_cast<float4*>(prev_input_px),
                    reinterpret_cast<float4*>(truncated_px), 
                    mask_px, dim, threshold);
            } else if (vectorization == 2) {
                active = activate_truncate_max_vectorized<float2, activation, 2>(
                    reinterpret_cast<float2*>(delta_px), 
                    reinterpret_cast<float2*>(prev_input_px),
                    reinterpret_cast<float2*>(truncated_px), 
                    mask_px, dim, threshold);
            } else {
                active = activate_truncate_max<scalar_t, activation>(delta_px, prev_input_px, truncated_px, mask_px, dim, threshold);
            }
        } else if (truncation_mode <= 2) {
            // euclidian norm / RMSE > threshold
            active = activate_truncate_sum<scalar_t, activation>(delta_px, prev_input_px, truncated_px, mask_px, dim, threshold, truncation_mode);
        }
        
#ifdef ENABLE_METRICS
        if (threadIdx.x % WARP_SIZE == 0) {
            int vals_read = truncation_mode == 0 ? 2 : 3;
            atomicAdd(&d_metrics->n_vals_read, uint64_t(vals_read * dim.in.c)); 
            int vals_written = truncation_mode == 0 ? 2 : (active ? 3 : 2);
            atomicAdd(&d_metrics->n_vals_written, uint64_t(vals_written * dim.in.c)); 
        }
#endif
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32>
__global__ void deltacnn_prepare_diff_mask_kernel(scalar_t * __restrict__ input, scalar_t * __restrict__ prev_input, scalar_t * __restrict__ delta_out, uint32_t *mask, float threshold, Dimensions dim) {
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
        scalar_t *input_px = &input[px_idx * dim.in.c];
        scalar_t *prev_input_px = &prev_input[px_idx * dim.in.c];
        scalar_t *delta_px = &delta_out[px_idx * dim.in.c];
        bool pixel_active = false;
        int first_pixel_active = -1;

        for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE) {
            int c = c_off + lane_idx;
            bool valid = c < dim.in.c;
            scalar_t in;
            scalar_t prev_in;
            scalar_t delta = 0.0f;
            if (valid) {
                in = input_px[c];
                prev_in = prev_input_px[c];
                delta = in - prev_in;
            }

            if (!pixel_active) {
                pixel_active |= fabsf(delta) > threshold;
                pixel_active = __any_sync(FULL_MASK, pixel_active);
                if (pixel_active) {
                    first_pixel_active = c;
                }
            }
            if (pixel_active && valid) {
                delta_px[c] = delta;
                prev_input_px[c] = in;
            }
        }

        for (int c = lane_idx; c < first_pixel_active && c < dim.in.c; c += WARP_SIZE) {
            scalar_t in = input_px[c];
            scalar_t prev_in = prev_input_px[c];
            scalar_t delta = in - prev_in;
            delta_px[c] = delta;
            prev_input_px[c] = in;
        }

        if (lane_idx == 0) {
            mask[px_idx] = pixel_active ? 1 : 0;
#ifdef ENABLE_METRICS
            atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
            // this would not be required in dense mode, don't count it!
            // atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.c)); 
            // atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.in.c)); 
            if (pixel_active) {
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
            }
#endif
        }
    }
}


template <typename scalar_t = float, int BLOCK_SIZE=128, int activation>
__device__ void aggregate_inputs_half2_no_truncation(scalar_t *delta_px, scalar_t *prev_input_px, uint32_t *mask_px, Dimensions dim, float threshold)
{
    const int lane_idx = threadIdx.x % WARP_SIZE;
    
    for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE*2) {
        int c = c_off + lane_idx*2;
        half2 delta;
        half2 sum;

        if (c < dim.in.c) {
            delta = *reinterpret_cast<half2*>(&delta_px[c]);
            half2 prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
            sum = __hadd2(delta, prev_in);
        
            delta = __hsub2(hactivation_selector2<activation>(sum), hactivation_selector2<activation>(prev_in));
        } else {
            delta = __float2half2_rn(0.0f);
        }

        if (c < dim.in.c) {
            *reinterpret_cast<half2*>(&prev_input_px[c]) = sum;
            if (activation > 0) {
                *reinterpret_cast<half2*>(&delta_px[c]) = delta;
            }
        }
    }
}

template<typename scalar_t = half, int BLOCK_SIZE=128, int activation>
__device__ bool aggregate_inputs_half2_max_truncation(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, float threshold, Dimensions dim){
    bool pixel_active = false;
    int pixel_active_detected = -999;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    
    for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE*2) {
        int c = c_off + lane_idx*2;
        half2 delta;
        half2 sum;
        half2 trunc;

        if (c < dim.in.c) {
            delta = *reinterpret_cast<half2*>(&delta_px[c]);
            half2 prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
            trunc = *reinterpret_cast<half2*>(&truncated_px[c]);
            trunc = __hadd2(delta, trunc);
            sum = __hadd2(trunc, prev_in);
        
            delta = __hsub2(hactivation_selector2<activation>(sum), hactivation_selector2<activation>(prev_in));
        } else {
            delta = __float2half2_rn(0.0f);
        }

        if (!pixel_active) {                
            // inverting the instruction here sind hblt2 returns true only if both are true
            pixel_active |= !__hble2(__habs2(delta), __float2half2_rn(threshold));
            pixel_active = __any_sync(FULL_MASK, pixel_active);
            if (pixel_active) {
                pixel_active_detected = c;
            }
        }

        if (pixel_active) {
            if (c < dim.in.c) {
                *reinterpret_cast<half2*>(&prev_input_px[c]) = sum;
                *reinterpret_cast<half2*>(&truncated_px[c]) = __float2half2_rn(0.0f);
                if (activation > 0) {
                    *reinterpret_cast<half2*>(&delta_px[c]) = delta;
                }
            }
        } else {
            if (c < dim.in.c) { 
                *reinterpret_cast<half2*>(&truncated_px[c]) = trunc;
            }
        }
    }
    for (int c = lane_idx * 2; c < pixel_active_detected && c < dim.in.c; c += WARP_SIZE*2) {
        half2 delta;
        half2 prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
        half2 trunc = *reinterpret_cast<half2*>(&truncated_px[c]);
        half2 sum = __hadd2(trunc, prev_in);
        
        if (activation != 0) {
            delta = __hsub2(hactivation_selector2<activation>(sum), hactivation_selector2<activation>(prev_in));
            *reinterpret_cast<half2*>(&delta_px[c]) = delta;
        }

        *reinterpret_cast<half2*>(&prev_input_px[c]) = sum;
        *reinterpret_cast<half2*>(&truncated_px[c]) = __float2half2_rn(0.0f);
    }


    if (lane_idx == 0 && !pixel_active) {
        *mask_px = 0;
    }
    return pixel_active;
}


template<typename scalar_t = half, int BLOCK_SIZE=128, int activation>
__device__ bool aggregate_inputs_half2_sum_truncation(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, float threshold, Dimensions dim, int truncation_mode) {
    const int lane_idx = threadIdx.x % WARP_SIZE;
    bool pixel_active = false;

    half2 pixel_sum = __float2half2_rn(0.0f);
    
    for (int c = lane_idx * 2; c < dim.in.c; c += WARP_SIZE*2) {
        half2 delta = *reinterpret_cast<half2*>(&delta_px[c]);
        half2 prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
        half2 trunc = *reinterpret_cast<half2*>(&truncated_px[c]);
        trunc = __hadd2(delta, trunc);
        half2 sum = __hadd2(trunc, prev_in);
    
        delta = __hsub2(hactivation_selector2<activation>(sum), hactivation_selector2<activation>(prev_in));
        pixel_sum = __hfma2(delta, delta, pixel_sum);
        *reinterpret_cast<half2*>(&truncated_px[c]) = trunc;
    }

    pixel_sum = Utils::warpReduceSum(pixel_sum);
    half single_pixel_sum = __hadd(__low2half(pixel_sum), __high2half(pixel_sum));
    if (truncation_mode == 1) {
        // euclidian norm
        pixel_active = __hgt(hsqrt(single_pixel_sum), __float2half(threshold));
    } else if (truncation_mode == 2) {
        // rmse
        pixel_active = __hgt(hsqrt(__hdiv(single_pixel_sum, __float2half(float(dim.in.c)))), __float2half(threshold));
    } else {
        pixel_active = true;
        if (threadIdx.x == 0) {
            printf("Truncation mode %i not implemented\n", truncation_mode);
        }
    }
    pixel_active = __shfl_sync(FULL_MASK, pixel_active, 0);

    for (int c = lane_idx * 2; c < dim.in.c; c += WARP_SIZE*2) {
        half2 prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
        half2 trunc = *reinterpret_cast<half2*>(&truncated_px[c]);
        half2 sum = __hadd2(trunc, prev_in);
        
        if (activation != 0) {
            half2 delta = __hsub2(hactivation_selector2<activation>(sum), hactivation_selector2<activation>(prev_in));
            *reinterpret_cast<half2*>(&delta_px[c]) = delta;
        }

        *reinterpret_cast<half2*>(&prev_input_px[c]) = sum;
        *reinterpret_cast<half2*>(&truncated_px[c]) = __float2half2_rn(0.0f);
    }


    if (lane_idx == 0 && !pixel_active) {
        *mask_px = 0;
    }
    return pixel_active;
}




template <typename scalar_t = half, int BLOCK_SIZE=128, int activation>
__device__ void aggregate_inputs_half_no_truncation(scalar_t *delta_px, scalar_t *prev_input_px, uint32_t *mask_px, Dimensions dim, float threshold)
{
    const int lane_idx = threadIdx.x % WARP_SIZE;
    
    for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE) {
        int c = c_off + lane_idx;
        half delta;
        half sum;

        if (c < dim.in.c) {
            delta = delta_px[c];
            half prev_in = prev_input_px[c];
            sum = __hadd(delta, prev_in);
        
            delta = __hsub(hactivation_selector<activation>(sum), hactivation_selector<activation>(prev_in));
        } else {
            delta = __float2half_rn(0.0f);
        }

        if (c < dim.in.c) {
            prev_input_px[c] = sum;
            if (activation > 0) {
                delta_px[c] = delta;
            }
        }
    }
}

template<typename scalar_t = half, int BLOCK_SIZE=128, int activation>
__device__ bool aggregate_inputs_half_max_truncation(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, float threshold, Dimensions dim) {
    const int lane_idx = threadIdx.x % WARP_SIZE;
    bool pixel_active = false;
    int pixel_active_detected = -999;

    for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE) {
        int c = c_off + lane_idx;
        scalar_t delta;
        scalar_t sum;
        scalar_t trunc;
        
        if (c < dim.in.c) {
            delta = delta_px[c];
            trunc = truncated_px[c];
            scalar_t prev_in = prev_input_px[c];
            trunc = __hadd(delta, trunc);
            sum = __hadd(trunc, prev_in);
            
            delta = __hsub(hactivation_selector<activation>(sum), hactivation_selector<activation>(prev_in));
        } else {
            delta = __float2half(0.0f);
        }

        if (!pixel_active) {
            pixel_active |= __hgt(__habs(delta), __float2half(threshold));
            pixel_active = __any_sync(FULL_MASK, pixel_active);
            if (pixel_active) {
                pixel_active_detected = c;
            }
        }
        if (pixel_active) {
            if (c < dim.in.c) {
                prev_input_px[c] = sum;
                truncated_px[c] = __float2half(0.0f);
                if (activation > 0) {
                    delta_px[c] = delta;
                }
            }
        } else {
            if (c < dim.in.c) {
                truncated_px[c] = trunc;
            }
        }
    }

    if (pixel_active) {
        for (int c = lane_idx; c < pixel_active_detected && c < dim.in.c; c += WARP_SIZE) {
            scalar_t prev_in = prev_input_px[c];
            scalar_t trunc = truncated_px[c];
            scalar_t sum = __hadd(trunc, prev_in);
            scalar_t delta;
            
            if (activation != 0) {
                delta = __hsub(hactivation_selector<activation>(sum), hactivation_selector<activation>(prev_in));
                delta_px[c] = delta;
            }
            prev_input_px[c] = sum;
            truncated_px[c] = __float2half(0.0f);
        }
    }
    else if (lane_idx == 0) {
        *mask_px = 0;
    }
    return pixel_active;
}




template<typename scalar_t = half, int BLOCK_SIZE=128, int activation>
__device__ bool aggregate_inputs_half_sum_truncation(scalar_t *delta_px, scalar_t *prev_input_px, scalar_t *truncated_px, uint32_t *mask_px, float threshold, Dimensions dim, int truncation_mode) {
    const int lane_idx = threadIdx.x % WARP_SIZE;
    bool pixel_active = false;

    half pixel_sum = __float2half(0.0f);
    
    for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
        half delta = delta_px[c];
        half prev_in = prev_input_px[c];
        half trunc = truncated_px[c];
        trunc = __hadd(delta, trunc);
        half sum = __hadd(trunc, prev_in);
    
        delta = __hsub(hactivation_selector<activation>(sum), hactivation_selector<activation>(prev_in));
        pixel_sum = __hfma(delta, delta, pixel_sum);
        truncated_px[c] = trunc;
    }

    pixel_sum = Utils::warpReduceSum(pixel_sum);
    if (truncation_mode == 1) {
        // euclidian norm
        pixel_active = __hgt(hsqrt(pixel_sum), __float2half(threshold));
    } else if (truncation_mode == 2) {
        // rmse
        pixel_active = __hgt(hsqrt(__hdiv(pixel_sum, __float2half(float(dim.in.c)))), __float2half(threshold));
    } else {
        pixel_active = true;
        if (threadIdx.x == 0) {
            printf("Truncation mode %i not implemented\n", truncation_mode);
        }
    }
    pixel_active = __shfl_sync(FULL_MASK, pixel_active, 0);

    if (pixel_active) {
        for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
            half prev_in = prev_input_px[c];
            half trunc = truncated_px[c];
            half sum = __hadd(trunc, prev_in);
            
            if (activation != 0) {
                half delta = __hsub(hactivation_selector<activation>(sum), hactivation_selector<activation>(prev_in));
                delta_px[c] = delta;
            }

            prev_input_px[c] = sum;
            truncated_px[c] = __float2half(0.0f);
        }
    }
    else if (lane_idx == 0) {
        *mask_px = 0;
    }
    return pixel_active;
}


template<typename scalar_t = half, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, int activation>
__global__ void deltacnn_activate_truncate_hp_kernel(scalar_t * __restrict__ delta, scalar_t * __restrict__ prev_input, scalar_t * __restrict__ truncated, uint32_t *mask, float threshold, Dimensions dim, int truncation_mode) {
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

    if (!checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask)) {
        return;
    }

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
#ifdef ENABLE_METRICS
        if (threadIdx.x % WARP_SIZE == 0) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.in.c)); 
        }
#endif
        if (mask[px_idx] == 0) 
            continue;

        scalar_t *delta_px = &delta[px_idx * dim.in.c];
        scalar_t *prev_input_px = &prev_input[px_idx * dim.in.c];
        scalar_t *truncated_px = &truncated[px_idx * dim.in.c];
        uint32_t *mask_px = &mask[px_idx];

        [[maybe_unused]] bool active = true;

        if (dim.in.c % 2 == 0) {
            if (truncation_mode < 0) {
                // no truncation
                aggregate_inputs_half2_no_truncation<scalar_t, BLOCK_SIZE, activation>(delta_px, prev_input_px, mask_px, dim, threshold);
            }
            else if (truncation_mode == 0) {
                active = aggregate_inputs_half2_max_truncation<half, BLOCK_SIZE, activation>(delta_px, prev_input_px, truncated_px, mask_px, threshold, dim);
            }
            else if (truncation_mode <= 2) {
                active = aggregate_inputs_half2_sum_truncation<half, BLOCK_SIZE, activation>(delta_px, prev_input_px, truncated_px, mask_px, threshold, dim, truncation_mode);
            }
        } else {
            if (truncation_mode < 0) {
                // no truncation
                aggregate_inputs_half_no_truncation<scalar_t, BLOCK_SIZE, activation>(delta_px, prev_input_px, mask_px, dim, threshold);
            }
            else if (truncation_mode == 0) {
                active = aggregate_inputs_half_max_truncation<half, BLOCK_SIZE, activation>(delta_px, prev_input_px, truncated_px, mask_px, threshold, dim);
            }
            else if (truncation_mode <= 2) {
                active = aggregate_inputs_half_sum_truncation<half, BLOCK_SIZE, activation>(delta_px, prev_input_px, truncated_px, mask_px, threshold, dim, truncation_mode);
            }
        }
        
#ifdef ENABLE_METRICS
            if (threadIdx.x % WARP_SIZE == 0) {
                int vals_read = truncation_mode == 0 ? 2 : 3;
                atomicAdd(&d_metrics->n_vals_read, uint64_t(vals_read * dim.in.c)); 
                int vals_written = truncation_mode == 0 ? 2 : (active ? 3 : 2);
                atomicAdd(&d_metrics->n_vals_written, uint64_t(vals_written * dim.in.c)); 
            }
#endif
    }
}

template<typename scalar_t = half, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32>
__global__ void deltacnn_prepare_diff_mask_hp_kernel(scalar_t * __restrict__ input, scalar_t * __restrict__ prev_input, scalar_t * __restrict__ delta_out, uint32_t *mask, float threshold, Dimensions dim) {
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

    if (dim.in.c % 2 == 0) {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            scalar_t *input_px = &input[px_idx * dim.in.c];
            scalar_t *prev_input_px = &prev_input[px_idx * dim.in.c];
            scalar_t *delta_px = &delta_out[px_idx * dim.in.c];
            bool pixel_active = false;
            int first_pixel_active = -1;

            for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE*2) {
                int c = c_off + lane_idx*2;
                bool valid = c < dim.in.c;
                half2 in;
                half2 prev_in;
                half2 delta = __float2half2_rn(0.0f);
                if (c < dim.in.c) {
                    in = *reinterpret_cast<half2*>(&input_px[c]);
                    prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
                    delta = __hsub2(in, prev_in);
                }

                if (!pixel_active) {
                    pixel_active |= !__hblt2(__habs2(delta), __float2half2_rn(threshold));
                    pixel_active = __any_sync(FULL_MASK, pixel_active);
                    if (pixel_active) {
                        first_pixel_active = c;
                    }
                }
                if (pixel_active && valid) {
                    *reinterpret_cast<half2*>(&delta_px[c]) = delta;
                    *reinterpret_cast<half2*>(&prev_input_px[c]) = in;
                }
            }

            for (int c = lane_idx*2; c < first_pixel_active; c += WARP_SIZE*2) {
                half2 in = *reinterpret_cast<half2*>(&input_px[c]);
                half2 prev_in = *reinterpret_cast<half2*>(&prev_input_px[c]);
                half2 delta = __hsub2(in, prev_in);
                *reinterpret_cast<half2*>(&delta_px[c]) = delta;
                *reinterpret_cast<half2*>(&prev_input_px[c]) = in;
            }

            if (lane_idx == 0) {
                mask[px_idx] = pixel_active ? 1 : 0;
#ifdef ENABLE_METRICS
            atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.in.c)); 
            if (pixel_active) {
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
            }
#endif
            }
        }
    }
    else {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            scalar_t *input_px = &input[px_idx * dim.in.c];
            scalar_t *prev_input_px = &prev_input[px_idx * dim.in.c];
            scalar_t *delta_px = &delta_out[px_idx * dim.in.c];
            bool pixel_active = false;
            int first_pixel_active = -1;

            for (int c_off = 0; c_off < dim.in.c; c_off += WARP_SIZE) {
                int c = c_off + lane_idx;
                bool valid = c < dim.in.c;
                scalar_t in;
                scalar_t prev_in;
                scalar_t delta = __float2half(0.0f);
                if (c < dim.in.c) {
                    in = input_px[c];
                    prev_in = prev_input_px[c];
                    delta = __hsub(in, prev_in);
                }

                if (!pixel_active) {
                    pixel_active |= __hgt(__habs(delta), __float2half(threshold));
                    pixel_active = __any_sync(FULL_MASK, pixel_active);
                    if (pixel_active) {
                        first_pixel_active = c;
                    }
                }
                if (pixel_active && valid) {
                    delta_px[c] = delta;
                    prev_input_px[c] = in;
                }
            }

            for (int c = lane_idx; c < first_pixel_active; c += WARP_SIZE) {
                scalar_t in = input_px[c];
                scalar_t prev_in = prev_input_px[c];
                scalar_t delta = __hsub(in, prev_in);
                delta_px[c] = delta;
                prev_input_px[c] = in;
            }

            if (lane_idx == 0) {
                mask[px_idx] = pixel_active ? 1 : 0;
#ifdef ENABLE_METRICS
                atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.c)); 
                atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.in.c)); 
                if (pixel_active) {
                    atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                }
#endif
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, int activation, bool dense_out>
__global__ void deltacnn_sparse_add_tensors_kernel(
        const scalar_t * __restrict__ val_a, 
        const scalar_t * __restrict__ val_b, 
        scalar_t * __restrict__ prev_out, 
        scalar_t * __restrict__ val_out, 
        uint32_t *mask_a, 
        uint32_t *mask_b, 
        uint32_t *mask_out,
        scalar_t weight_a,
        scalar_t weight_b,
        Dimensions dim
        ) 
{
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(2 * (end_idx-start_idx) * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx-start_idx) * dim.in.c)); 
    }
#endif

    if (!checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask_a) &&
        !checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask_b)
    ) {
        for (int px_idx = start_idx + threadIdx.x; px_idx < end_idx; px_idx += BLOCK_SIZE) {
            mask_out[px_idx] = 0;
        }
        if (dense_out) {
            for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
                scalar_t *out_px = &val_out[px_idx * dim.in.c];
                scalar_t *prev_out_px = prev_out == nullptr ? nullptr : &prev_out[px_idx * dim.in.c];
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    scalar_t prev_out = prev_out_px[c];
                    
                    prev_out = activation_selector(prev_out, activation);

                    out_px[c] = prev_out;
                }
            }
        }
        return;
    }

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
        bool active_a = mask_a[px_idx] != 0;
        bool active_b = mask_b[px_idx] != 0;

        scalar_t *out_px = &val_out[px_idx * dim.in.c];
        scalar_t *prev_out_px = prev_out == nullptr ? nullptr : &prev_out[px_idx * dim.in.c];

        if (!(active_a || active_b)) {
            if (lane_idx == 0 && !dense_out) {
                mask_out[px_idx] = 0;
            }
            if (dense_out) {
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    scalar_t prev_out = prev_out_px[c];
                    
                    prev_out = activation_selector(prev_out, activation);

                    out_px[c] = prev_out;
                }
            }
            continue;
        }

#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
        }
#endif
        
        if (lane_idx == 0 && !dense_out) {
            mask_out[px_idx] = 1;
        }

        const scalar_t *a_px = &val_a[px_idx * dim.in.c];
        const scalar_t *b_px = &val_b[px_idx * dim.in.c];

        if (dim.in.c % 4 == 0 && dim.in.c >= 128) {
            for(int c = lane_idx * 4; c+3 < dim.in.c; c += WARP_SIZE*4) {
                scalar_t sum[4];

                if (active_a && active_b) {
                    float b_vals[4];
                    *reinterpret_cast<float4*>(&sum[0]) = *reinterpret_cast<const float4*>(&a_px[c]);
                    *reinterpret_cast<float4*>(&b_vals[0]) = *reinterpret_cast<const float4*>(&b_px[c]);

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] = sum[i] * weight_a + b_vals[i] * weight_b;
                    }
                } else if (active_a) {
                    *reinterpret_cast<float4*>(&sum[0]) = *reinterpret_cast<const float4*>(&a_px[c]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] *= weight_a;
                    }
                } else {
                    *reinterpret_cast<float4*>(&sum[0]) = *reinterpret_cast<const float4*>(&b_px[c]);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] *= weight_b;
                    }
                }

                if (dense_out) {
                    scalar_t prev_out_val[4];
                    *reinterpret_cast<float4*>(&prev_out_val[0]) = *reinterpret_cast<float4*>(&prev_out_px[c]); 
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] += prev_out_val[i];
                        prev_out_px[c+i] = sum[i];
                            
                        sum[i] = activation_selector<activation>(sum[i]);
                    }
                    *reinterpret_cast<float4*>(&out_px[c]) = *reinterpret_cast<float4*>(&sum[0]);
                } else {
                    if (activation <= 0) {
                        *reinterpret_cast<float4*>(&out_px[c]) = *reinterpret_cast<float4*>(&sum[0]);
                        continue;
                    }


                    scalar_t prev_out_val[4];
                    *reinterpret_cast<float4*>(&prev_out_val[0]) = *reinterpret_cast<float4*>(&prev_out_px[c]); 
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] += prev_out_val[i];
                        prev_out_px[c+i] = sum[i];
                        
                        sum[i] = activation_selector<activation>(sum[i]) - activation_selector<activation>(prev_out_val[i]);
                    }
                    *reinterpret_cast<float4*>(&out_px[c]) = *reinterpret_cast<float4*>(&sum[0]);
                }
            }
        } else {
            for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                scalar_t sum = 0.0f;
                
                if (active_a) {
                    sum = a_px[c] * weight_a;
                }
                if (active_b) {
                    sum += b_px[c] * weight_b;
                }

                if (dense_out) {
                    scalar_t prev_out_val = prev_out_px[c]; 
                    sum += prev_out_val;
                    prev_out_px[c] = sum;
                        
                    sum = activation_selector<activation>(sum);
                    out_px[c] = sum;

                } else {
                    if (activation <= 0) {
                        out_px[c] = sum;
                        continue;
                    }

                    scalar_t prev_out_val = prev_out_px[c]; 
                    sum += prev_out_val;
                    prev_out_px[c] = sum;
                    scalar_t delta; 
                    
                    delta = activation_selector<activation>(sum) - activation_selector<activation>(prev_out_val);

                    out_px[c] = delta;
                }
            }
        }
    }
}


template<typename scalar_t = half, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32>
__global__ void deltacnn_sparse_add_tensors_hp_kernel(
    const scalar_t * __restrict__ val_a, 
    const scalar_t * __restrict__ val_b, 
    scalar_t * __restrict__ prev_out, 
    scalar_t * __restrict__ val_out, 
    uint32_t *mask_a, 
    uint32_t *mask_b, 
    uint32_t *mask_out, 
    float weight_a, 
    float weight_b,
    Dimensions dim, 
    int activation,
    bool dense_out
    ) 
{
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

    const half2 weight_a_h2 = __float2half2_rn(weight_a);
    const half2 weight_b_h2 = __float2half2_rn(weight_b);
    const half weight_a_h = __float2half(weight_a);
    const half weight_b_h = __float2half(weight_b);

    // TODO implement early termination if masks are completely sparse!
    
#ifdef ENABLE_METRICS
    if (threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(2 * (end_idx-start_idx) * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx-start_idx) * dim.in.c)); 
    }
#endif

    if (dim.in.c % 2 == 0) {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active_a = mask_a[px_idx] != 0;
            bool active_b = mask_b[px_idx] != 0;

            scalar_t *out_px = &val_out[px_idx * dim.in.c];
            scalar_t *prev_out_px = prev_out == nullptr ? nullptr : &prev_out[px_idx * dim.in.c];

            if (!(active_a || active_b)) {
                if (lane_idx == 0 && !dense_out) {
                    mask_out[px_idx] = 0;
                }
                if (dense_out) {
                    for (int c = lane_idx*2; c < dim.in.c; c += WARP_SIZE*2) {
                        half2 prev_out = *reinterpret_cast<const half2*>(&prev_out_px[c]);
                        prev_out = hactivation_selector2(prev_out, activation);
                        *reinterpret_cast<half2*>(&out_px[c]) = prev_out;
                    }
                }
                continue;
            }

#ifdef ENABLE_METRICS
            if (lane_idx == 0) {
                atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
            }
#endif
            
            if (lane_idx == 0 && !dense_out) {
                mask_out[px_idx] = 1;
            }

            const scalar_t *a_px = &val_a[px_idx * dim.in.c];
            const scalar_t *b_px = &val_b[px_idx * dim.in.c];
            for (int c = lane_idx*2; c < dim.in.c; c += WARP_SIZE*2) {
                half2 sum = __float2half2_rn(0.0f);
                
                if (active_a) {
                    sum = __hmul2(*reinterpret_cast<const half2*>(&a_px[c]), weight_a_h2);
                }
                if (active_b) {
                    sum = __hfma2(*reinterpret_cast<const half2*>(&b_px[c]), weight_b_h2, sum);
                }

                if (dense_out) {
                    half2 prev_out_val = *reinterpret_cast<half2*>(&prev_out_px[c]);
                    sum = __hadd2(sum, prev_out_val);
                    *reinterpret_cast<half2*>(&prev_out_px[c]) = sum;
                    sum = hactivation_selector2(sum, activation);
                    *reinterpret_cast<half2*>(&out_px[c]) = sum;

                } else {
                    if (activation <= 0) {
                        *reinterpret_cast<half2*>(&out_px[c]) = sum;
                        continue;
                    }

                    half2 prev_out_val = *reinterpret_cast<half2*>(&prev_out_px[c]); 
                    sum = __hadd2(sum, prev_out_val);
                    *reinterpret_cast<half2*>(&prev_out_px[c]) = sum;
                    half2 delta = __hsub2(hactivation_selector2(sum, activation), hactivation_selector2(prev_out_val, activation));
                    *reinterpret_cast<half2*>(&out_px[c]) = delta;
                }
            }
        }

    } else {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active_a = mask_a[px_idx] != 0;
            bool active_b = mask_b[px_idx] != 0;

            scalar_t *out_px = &val_out[px_idx * dim.in.c];
            scalar_t *prev_out_px = prev_out == nullptr ? nullptr : &prev_out[px_idx * dim.in.c];

            if (!(active_a || active_b)) {
                if (lane_idx == 0 && !dense_out) {
                    mask_out[px_idx] = 0;
                }
                if (dense_out) {
                    for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                        scalar_t prev_out = prev_out_px[c];
                        prev_out = hactivation_selector(prev_out, activation);
                        out_px[c] = prev_out;
                    }
                }
                continue;
            }
            
            if (lane_idx == 0 && !dense_out) {
                mask_out[px_idx] = 1;
            }

            const scalar_t *a_px = &val_a[px_idx * dim.in.c];
            const scalar_t *b_px = &val_b[px_idx * dim.in.c];
            for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                scalar_t sum = __float2half(0.0f);
                
                if (active_a) {
                    sum = __hmul(a_px[c], weight_a_h);
                }
                if (active_b) {
                    sum = __hfma(b_px[c], weight_b_h, sum);
                }

                if (dense_out) {
                    scalar_t prev_out_val = prev_out_px[c]; 
                    sum = __hadd(sum, prev_out_val);
                    prev_out_px[c] = sum;
                    sum = hactivation_selector(sum, activation);
                    out_px[c] = sum;

                } else {
                    if (activation <= 0) {
                        out_px[c] = sum;
                        continue;
                    }

                    scalar_t prev_out_val = prev_out_px[c]; 
                    sum = __hadd(sum, prev_out_val);
                    prev_out_px[c] = sum;
                    scalar_t delta; 
                        
                    delta = __hsub(hactivation_selector(sum, activation), hactivation_selector(prev_out_val, activation));

                    out_px[c] = delta;
                }
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, bool USE_ACTIVATION>
__global__ void deltacnn_sparse_add_to_dense_tensor_kernel_sp(
        scalar_t * __restrict__ val_a, 
        scalar_t * __restrict__ val_b, 
        uint32_t *mask_a, 
        Dimensions dim,
        int activation
        ) 
{
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(2 * (end_idx-start_idx) * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx-start_idx) * dim.in.c)); 
    }
#endif

    if (!USE_ACTIVATION && !checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask_a)) {
        return;
    }

    if (dim.in.c % 4 == 0 && dim.in.c >= 128) {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active_a = mask_a[px_idx] != 0;

            if (!active_a) {
                if (!USE_ACTIVATION) {
                    continue;
                }
                else {
                    scalar_t *a_px = &val_a[px_idx * dim.in.c];
                    scalar_t *b_px = &val_b[px_idx * dim.in.c];
                    for (int c = lane_idx*4; c+3 < dim.in.c; c += WARP_SIZE*4) {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            a_px[c+i] = activation_selector<scalar_t>(b_px[c+i], activation);                    
                        }
                    }
#ifdef ENABLE_METRICS
                    if (lane_idx == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                        atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                    }
#endif
                }
            } else {
                scalar_t *a_px = &val_a[px_idx * dim.in.c];
                scalar_t *b_px = &val_b[px_idx * dim.in.c];
                for (int c = lane_idx*4; c+3 < dim.in.c; c += WARP_SIZE*4) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        if (USE_ACTIVATION) {
                            float result = b_px[c+i] + a_px[c+i];
                            b_px[c+i] = result;
                            a_px[c+i] = activation_selector<scalar_t>(result, activation);
                        }
                        else {
                            b_px[c+i] += a_px[c+i];
                        }             
                    }
                }
#ifdef ENABLE_METRICS
                if (lane_idx == 0) {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                }
#endif
            }
        }
    } else {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active_a = mask_a[px_idx] != 0;

            if (!active_a) {
                if (!USE_ACTIVATION)
                    continue;
                else {
                    scalar_t *a_px = &val_a[px_idx * dim.in.c];
                    scalar_t *b_px = &val_b[px_idx * dim.in.c];
                    for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                        a_px[c] = activation_selector<scalar_t>(b_px[c], activation);    
                    }
#ifdef ENABLE_METRICS
                    if (lane_idx == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                        atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                    }
#endif
                }
            } else  {
                scalar_t *a_px = &val_a[px_idx * dim.in.c];
                scalar_t *b_px = &val_b[px_idx * dim.in.c];
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    if (USE_ACTIVATION) {
                        float result = b_px[c] + a_px[c];
                        b_px[c] = result;
                        a_px[c] = activation_selector<scalar_t>(result, activation);
                    }
                    else {
                        b_px[c] += a_px[c];
                    }     
                }
#ifdef ENABLE_METRICS
                if (lane_idx == 0) {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                }
#endif
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, bool USE_ACTIVATION>
__global__ void deltacnn_sparse_add_to_dense_tensor_kernel_hp(
        scalar_t * __restrict__ val_a, 
        scalar_t * __restrict__ val_b, 
        uint32_t *mask_a, 
        Dimensions dim,
        int activation
        ) 
{
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(2 * (end_idx-start_idx) * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx-start_idx) * dim.in.c)); 
    }
#endif

    if (!USE_ACTIVATION && !checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask_a)) {
        return;
    }

    if (dim.out.c % 2 == 0) {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active_a = mask_a[px_idx] != 0;

            if (!active_a) {
                if (!USE_ACTIVATION)
                    continue;
                else {
                    scalar_t *a_px = &val_a[px_idx * dim.in.c];
                    scalar_t *b_px = &val_b[px_idx * dim.in.c];
                    for (int c = lane_idx*2; c < dim.in.c; c += WARP_SIZE*2) {
                        *reinterpret_cast<half2*>(&a_px[c]) = hactivation_selector2(*reinterpret_cast<half2*>(&b_px[c]), activation);
                    }
#ifdef ENABLE_METRICS
                    if (lane_idx == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                        atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                    }
#endif
                }
            } else  {
                scalar_t *a_px = &val_a[px_idx * dim.in.c];
                scalar_t *b_px = &val_b[px_idx * dim.in.c];
                for (int c = lane_idx*2; c < dim.in.c; c += WARP_SIZE*2) {
                    if (USE_ACTIVATION) {
                        half2 result = __hadd2(*reinterpret_cast<half2*>(&a_px[c]), *reinterpret_cast<half2*>(&b_px[c]));
                        *reinterpret_cast<half2*>(&b_px[c]) = result;
                        *reinterpret_cast<half2*>(&a_px[c]) = hactivation_selector2(result, activation);
                    } else {
                        *reinterpret_cast<half2*>(&b_px[c]) = __hadd2(*reinterpret_cast<half2*>(&a_px[c]), *reinterpret_cast<half2*>(&b_px[c]));
                    }
                }
#ifdef ENABLE_METRICS
                if (lane_idx == 0) {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                }
#endif
            }
        }
    } else {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active_a = mask_a[px_idx] != 0;

            if (!active_a) {
                if (!USE_ACTIVATION)
                    continue;
                else {
                    scalar_t *a_px = &val_a[px_idx * dim.in.c];
                    scalar_t *b_px = &val_b[px_idx * dim.in.c];
                    for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                        a_px[c] = hactivation_selector(b_px[c], activation);
                    }
#ifdef ENABLE_METRICS
                    if (lane_idx == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                        atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                    }
#endif
                }
            } else  {
                scalar_t *a_px = &val_a[px_idx * dim.in.c];
                scalar_t *b_px = &val_b[px_idx * dim.in.c];
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    if (USE_ACTIVATION) {
                        half result = __hadd(a_px[c], b_px[c]);
                        b_px[c] = result;
                        a_px[c] = hactivation_selector(result, activation);
                    } else {
                        b_px[c] = __hadd(a_px[c], b_px[c]);
                    }
                }
#ifdef ENABLE_METRICS
                if (lane_idx == 0) {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
                }
#endif
            }
        }
    }
}



template<typename scalar_t, int POOL_MODE>
__device__ __forceinline__ void pooling_step(scalar_t& result, scalar_t& result_prev, scalar_t acc_val, scalar_t prev_in) {
    if (POOL_MODE == 0) {
        // MAX Pooling
        result = fmaxf(acc_val, result);
        result_prev = fmaxf(prev_in, result_prev);
    } else {
        // AVG Pooling
        result += acc_val;
    }
}



template<typename scalar_t, int POOL_MODE>
__device__ __forceinline__ void pooling_step_hp(scalar_t& result, scalar_t& result_prev, scalar_t acc_val, scalar_t prev_in) {
    if (POOL_MODE == 0) {
        // MAX Pooling
        // result = __hmax(acc_val, result);
        // result_prev = __hmax(prev_in, result_prev);
        // TODO check when __hmax was introduced and use it for later hardware
        result = __hgt(acc_val, result) ? acc_val : result;
        result_prev = __hgt(prev_in, result_prev) ? prev_in : result_prev;
    } else {
        // AVG Pooling
        result = __hadd(acc_val, result);
    }
}


template<typename scalar_t = float, int KERNEL_SIZE=3, int pixelsPerBlock=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, int DILATION=1, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int32_t blocksPerBatch = divup(dim.out.w, pixelsPerBlock*DILATION) * divup(dim.out.h, pixelsPerBlock*DILATION)*DILATION*DILATION;
    const uint32_t batch = blockIdx.x / blocksPerBatch;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlock * pixelsPerBlock;
    const int K_HALF = KERNEL_SIZE / 2;
    const int K_HALF_RIGHT = (KERNEL_SIZE-1) / 2;

    const int tile_idx = blockIdx.x % blocksPerBatch;
    const int tiles_per_x = divup(dim.out.w, pixelsPerBlock * DILATION);
    const int dilation_idx = tile_idx / (DILATION * DILATION);
    const int dilation_sub_idx = tile_idx % (DILATION * DILATION);
    const int tile_start_out_y = (dilation_idx / tiles_per_x) * pixelsPerBlock * DILATION + (dilation_sub_idx / DILATION);
    const int tile_start_out_x = (dilation_idx % tiles_per_x) * pixelsPerBlock * DILATION + (dilation_sub_idx % DILATION);
    const int tile_start_in_y = tile_start_out_y * STRIDE - config.padding[0];
    const int tile_start_in_x = tile_start_out_x * STRIDE - config.padding[1];
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;
    const int w_in = pixelsPerBlock + (pixelsPerBlock-1) * (STRIDE-1) + (K_HALF+K_HALF_RIGHT);
    const int n_in_px = w_in * w_in;

    __shared__ uint32_t s_mask[n_in_px];

    for (int i = threadIdx.x; i < n_in_px; i += blockDim.x) {
        int y = tile_start_in_y + (i / w_in) * DILATION;
        int x = tile_start_in_x + (i % w_in) * DILATION;
        if (y >= 0 && y < dim.in.h && x >= 0 && x < dim.in.w) {
            const int mask_idx = y * dim.in.w + x;
            s_mask[i] = batch_mask != nullptr ? batch_mask[mask_idx] : 1;
        } else {
            s_mask[i] = 0;
        }
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
    //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
    //     );
    // }

    int density = 0;
    uint64_t t_mask = 0LLU;
    for (int i = 0; i < n_in_px; ++i) {
        if (s_mask[i] != 0) {
            t_mask += (1LLU << i);
            ++density;
        }
    }

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.z == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
        if (POOL_MODE == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * 2 * dim.in.c)); 
        } else {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * dim.in.c)); 
        }
    }
#endif

    if (out_mask != nullptr && blockIdx.z == 0) {
        uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];
        for (int out_px = threadIdx.x; out_px < n_pixels_out; out_px += BLOCK_SIZE) {
            int p_y = out_px / pixelsPerBlock;
            int p_x = out_px % pixelsPerBlock;
            int out_y = p_y * DILATION + tile_start_out_y;
            int out_x = p_x * DILATION + tile_start_out_x; 
            if (out_y >= dim.out.h || out_x >= dim.out.w)
                continue;

            bool updated = false;
            for (int y = 0; y < KERNEL_SIZE; y++) {
                for (int x = 0; x < KERNEL_SIZE; x++) {
                    int in_px_idx = ((p_y*STRIDE) + y) * w_in + ((p_x*STRIDE) + x);
                    updated |= (t_mask & (1LLU << in_px_idx)) != 0;
                }
            }
            batch_out_mask[out_y * dim.out.w + out_x] = updated ? 1:0;
#ifdef ENABLE_METRICS
            if (updated) {
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
            }
#endif
        }
    }

    if (t_mask == 0LLU) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            for (int out_idx = threadIdx.x; out_idx < n_pixels_out * dim.out.c; out_idx += BLOCK_SIZE) {
                int out_px = out_idx / dim.out.c;
                int out_c = out_idx % dim.out.c;
                int p_y = out_px / pixelsPerBlock;
                int p_x = out_px % pixelsPerBlock;
                int out_y = p_y * DILATION + tile_start_out_y;
                int out_x = p_x * DILATION + tile_start_out_x; 
                if (out_y >= dim.out.h || out_x >= dim.out.w)
                    continue;
                batch_out[(out_y * dim.out.w + out_x) * dim.out.c + out_c] = 0.0f; 
            }
        }
        return;
    }

    const bool requires_boundary_checks = SUB_TILE_SPARSITY || (tile_start_in_x <= 0 || tile_start_in_y <= 0 || tile_start_in_x + pixelsPerBlock * STRIDE * DILATION + 2 >= dim.in.w || tile_start_in_y + pixelsPerBlock * STRIDE * DILATION + 2 >= dim.in.h); 

    for (int out_c = tile_start_z + threadIdx.x; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE) {
        scalar_t t_out[n_pixels_out];
        scalar_t t_out_prev[n_pixels_out];
        if (POOL_MODE == 0) {
            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                // TODO find a better way to find float max neg value
                t_out[i] = -1e30f;
                t_out_prev[i] = -1e-30f;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                t_out[i] = scalar_t(0.0f);
                t_out_prev[i] = scalar_t(0.0f);
            }
        }
        
        if (requires_boundary_checks) {
            #pragma unroll
            for (int in_y = -K_HALF; in_y < -K_HALF+w_in; ++in_y) {
                #pragma unroll
                for (int in_x = -K_HALF; in_x < -K_HALF+w_in; ++in_x) {
                    const int in_y_im = (in_y+K_HALF) * DILATION + tile_start_in_y;
                    const int in_x_im = (in_x+K_HALF) * DILATION + tile_start_in_x;
                    const bool inside = in_y_im >= 0 && in_y_im < dim.in.h && in_x_im >= 0 && in_x_im < dim.in.w;
                    if (inside) {
                        const bool valid = s_mask[((in_y+K_HALF) * w_in + (in_x+K_HALF))] != 0;
                        if (!valid && POOL_MODE == 1) {
                            // TODO improve like single pixel version
                            continue;
                        }
                        const int in_idx = (in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c;
                        const scalar_t delta = valid ? batch_in[in_idx] : scalar_t(0.0f);
                        const scalar_t prev_in = POOL_MODE == 0 ? batch_prev_in[in_idx] : 0.0f;
                        const scalar_t acc_val = delta + prev_in;

                        const int min_f_y = -in_y;
                        const int min_f_x = -in_x;

                        const int max_f_y = (-K_HALF+w_in-1) - in_y - K_HALF_RIGHT;
                        const int max_f_x = (-K_HALF+w_in-1) - in_x - K_HALF_RIGHT;


                        // this calculates the first index of the kernel that is used for processing. with strided convs/pools, not all filter indices will be used
                        const int stride_off_y = (((-in_y + K_HALF_RIGHT) % STRIDE) + STRIDE) % STRIDE;
                        const int stride_off_x = (((-in_x + K_HALF_RIGHT) % STRIDE) + STRIDE) % STRIDE;

                        // if (blockIdx.x == 0 && threadIdx.x == 0) {
                        //     printf("in_y=%i in_x=%i in_y_im=%i in_x_im=%i, s_off=[%i,%i], f_y=[%i, %i], f_x=[%i, %i], valid=%i, acc_val=%f, delta=%f, prev_in=%f\n",
                        //         in_y, in_x, in_y_im, in_x_im, stride_off_y, stride_off_x,
                        //         Utils::constexpr_max(-K_HALF_RIGHT + stride_off_y, min_f_y), Utils::constexpr_min(K_HALF, max_f_y),
                        //         Utils::constexpr_max(-K_HALF_RIGHT + stride_off_x, min_f_x), Utils::constexpr_min(K_HALF, max_f_x),
                        //         (valid?1:0), acc_val, delta, prev_in
                        //     );
                        // }

                        #pragma unroll
                        for (int f_y = Utils::constexpr_max(-K_HALF_RIGHT + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                            #pragma unroll
                            for (int f_x = Utils::constexpr_max(-K_HALF_RIGHT + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                // if (blockIdx.x == 0 && threadIdx.x == 0 && ((in_y+f_y)/STRIDE)==1 && ((in_x+f_x)/STRIDE)==1) {
                                //     printf("out_y=%i out_x=%i, f_y=%i f_x=%i, in_y=%i in_x=%i in_y_im=%i in_x_im=%i valid=%i, acc_val=%f, delta=%f, prev_in=%f, t_out=%f K_HALF_RIGHT=%i stride_off_y=%i, stride_off_x=%i\n",
                                //         ((in_y+f_y)/STRIDE), ((in_x+f_x)/STRIDE),
                                //         f_y, f_x, in_y, in_x, in_y_im, in_x_im, (valid?1:0), acc_val, delta, prev_in,
                                //         t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)], K_HALF_RIGHT,
                                //         stride_off_y, stride_off_x
                                //     );
                                // }
                                if (POOL_MODE == 0) {
                                    // MAX Pooling
                                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] = fmaxf(acc_val, t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)]);
                                    t_out_prev[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] = fmaxf(prev_in, t_out_prev[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)]);
                                } else {
                                    // AVG Pooling
                                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] += acc_val;
                                    t_out_prev[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] += prev_in;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int in_y = -K_HALF; in_y < pixelsPerBlock * STRIDE + K_HALF_RIGHT; ++in_y) {
                #pragma unroll
                for (int in_x = -K_HALF; in_x < pixelsPerBlock * STRIDE + K_HALF_RIGHT; ++in_x) {
                    const int in_y_im = (in_y+K_HALF) * DILATION + tile_start_in_y;
                    const int in_x_im = (in_x+K_HALF) * DILATION + tile_start_in_x;
                    const int in_idx = (in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c;
                    const scalar_t delta = batch_in[in_idx];
                    const scalar_t prev_in = POOL_MODE == 0 ? batch_prev_in[in_idx] : 0.0f;
                    const scalar_t acc_val = delta + prev_in;

                    const int min_f_y = -in_y;
                    const int min_f_x = -in_x;
                    const int max_f_y = pixelsPerBlock * STRIDE - in_y - 1;
                    const int max_f_x = pixelsPerBlock * STRIDE - in_x - 1;

                    const int stride_off_y = (-K_HALF - in_y - 1) % KERNEL_SIZE - (KERNEL_SIZE - STRIDE);
                    const int stride_off_x = (-K_HALF - in_x - 1) % KERNEL_SIZE - (KERNEL_SIZE - STRIDE);

                    #pragma unroll
                    for (int f_y = Utils::constexpr_max(-K_HALF_RIGHT + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                        #pragma unroll
                        for (int f_x = Utils::constexpr_max(-K_HALF_RIGHT + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                            // if (blockIdx.x == 0 && threadIdx.x == 0 && in_y+f_y == 0 && in_x+f_x == 0) {
                            //     printf("f_y=%i f_x=%i, in_y=%i in_x=%i in_y_im=%i in_x_im=%i acc_val=%f, delta=%f, prev_in=%f, t_out=%f K_HALF_RIGHT=%i\n",
                            //         f_y, f_x, in_y, in_x, in_y_im, in_x_im, acc_val, delta, prev_in,
                            //         t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)], K_HALF_RIGHT
                            //     );
                            // }
                            if (POOL_MODE == 0) {
                                // MAX Pooling
                                t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] = fmaxf(acc_val, t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)]);
                                t_out_prev[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] = fmaxf(prev_in, t_out_prev[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)]);
                            } else {
                                // AVG Pooling
                                t_out[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] += acc_val;
                                t_out_prev[((in_y+f_y)/STRIDE) * pixelsPerBlock + ((in_x+f_x)/STRIDE)] += prev_in;
                            }
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int out_y = 0; out_y < pixelsPerBlock; ++out_y) {
            const int out_y_im = out_y * DILATION + tile_start_out_y;
            #pragma unroll
            for (int out_x = 0; out_x < pixelsPerBlock; ++out_x) {
                const int out_x_im = out_x * DILATION + tile_start_out_x;
                const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                if (valid) {
                    if (POOL_MODE == 0) {
                        // MAX Pooling
                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = t_out[out_y*pixelsPerBlock + out_x] - t_out_prev[out_y*pixelsPerBlock + out_x];
                    } else {
                        // AVG Pooling
                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = t_out[out_y*pixelsPerBlock + out_x] / (KERNEL_SIZE * KERNEL_SIZE);
                    }
                }
            }
        }
    }
}


template<typename scalar_t = float, int KERNEL_SIZE=3, int pixelsPerBlock=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, int DILATION=1, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_less_registers(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int32_t blocksPerBatch = divup(dim.out.w, pixelsPerBlock*DILATION) * divup(dim.out.h, pixelsPerBlock*DILATION)*DILATION*DILATION;
    const uint32_t batch = blockIdx.x / blocksPerBatch;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlock * pixelsPerBlock;
    const int K_HALF = KERNEL_SIZE / 2;
    const int K_HALF_RIGHT = (KERNEL_SIZE-1) / 2;

    const int tile_idx = blockIdx.x % blocksPerBatch;
    const int tiles_per_x = divup(dim.out.w, pixelsPerBlock * DILATION);
    const int dilation_idx = tile_idx / (DILATION * DILATION);
    const int dilation_sub_idx = tile_idx % (DILATION * DILATION);
    const int tile_start_out_y = (dilation_idx / tiles_per_x) * pixelsPerBlock * DILATION + (dilation_sub_idx / DILATION);
    const int tile_start_out_x = (dilation_idx % tiles_per_x) * pixelsPerBlock * DILATION + (dilation_sub_idx % DILATION);
    const int tile_start_in_y = tile_start_out_y * STRIDE - config.padding[0];
    const int tile_start_in_x = tile_start_out_x * STRIDE - config.padding[1];
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;
    const int w_in = pixelsPerBlock + (pixelsPerBlock-1) * (STRIDE-1) + (K_HALF+K_HALF_RIGHT);
    const int n_in_px = w_in * w_in;

    __shared__ uint32_t s_mask[n_in_px];

    for (int i = threadIdx.x; i < n_in_px; i += blockDim.x) {
        int y = tile_start_in_y + (i / w_in) * DILATION;
        int x = tile_start_in_x + (i % w_in) * DILATION;
        if (y >= 0 && y < dim.in.h && x >= 0 && x < dim.in.w) {
            const int mask_idx = y * dim.in.w + x;
            s_mask[i] = batch_mask != nullptr ? batch_mask[mask_idx] : 1;
        } else {
            s_mask[i] = 0;
        }
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
    //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
    //     );
    // }

    int density = 0;
    uint64_t t_mask = 0LLU;
    for (int i = 0; i < n_in_px; ++i) {
        if (s_mask[i] != 0) {
            t_mask += (1LLU << i);
            ++density;
        }
    }

    

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
        if (POOL_MODE == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * 2 * dim.in.c)); 
        } else {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * dim.in.c)); 
        }
    }
#endif

    if (out_mask != nullptr) {
        uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];
        for (int out_px = threadIdx.x; out_px < n_pixels_out; out_px += BLOCK_SIZE) {
            int p_y = out_px / pixelsPerBlock;
            int p_x = out_px % pixelsPerBlock;
            int out_y = p_y * DILATION + tile_start_out_y;
            int out_x = p_x * DILATION + tile_start_out_x; 
            if (out_y >= dim.out.h || out_x >= dim.out.w)
                continue;

            bool updated = false;
            for (int y = 0; y < KERNEL_SIZE; y++) {
                for (int x = 0; x < KERNEL_SIZE; x++) {
                    int in_px_idx = ((p_y*STRIDE) + y) * w_in + ((p_x*STRIDE) + x);
                    
                    updated |= w_in > 8 ? 
                        (s_mask[in_px_idx] != 0) :
                        ((t_mask & (1LLU << in_px_idx)) != 0);
                }
            }
            batch_out_mask[out_y * dim.out.w + out_x] = updated ? 1:0;
#ifdef ENABLE_METRICS
            if (updated) {
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
            }
#endif
        }
    }

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            for (int out_idx = threadIdx.x; out_idx < n_pixels_out * dim.out.c; out_idx += BLOCK_SIZE) {
                int out_px = out_idx / dim.out.c;
                int out_c = out_idx % dim.out.c;
                int p_y = out_px / pixelsPerBlock;
                int p_x = out_px % pixelsPerBlock;
                int out_y = p_y * DILATION + tile_start_out_y;
                int out_x = p_x * DILATION + tile_start_out_x; 
                if (out_y >= dim.out.h || out_x >= dim.out.w)
                    continue;
                batch_out[(out_y * dim.out.w + out_x) * dim.out.c + out_c] = 0.0f; 
            }
        }

        // if (threadIdx.x == 0) {
        //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
        //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
        //     );
        // }
        return;
    }

    for (int out_c = tile_start_z + threadIdx.x; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE) {
        for (int out_y = 0; out_y < pixelsPerBlock && out_y + tile_start_out_y < dim.out.h; ++out_y) {
            for (int out_x = 0; out_x < pixelsPerBlock && out_x + tile_start_out_x < dim.out.w; ++out_x) {
                scalar_t result = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
                scalar_t result_prev = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
                
                for (int in_y = out_y * STRIDE; in_y < out_y * STRIDE + KERNEL_SIZE; ++in_y){
                    const int in_y_im = in_y + tile_start_in_y;
                    if (in_y_im < 0) {
                        // pooling_step<scalar_t, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                        continue;
                    } 
                    if (in_y_im >= dim.in.h) {
                        // this is just to get same behaviour as when first applying padding and then pooling
                        if (in_y_im < dim.in.h + config.padding[2] && config.padding[0] != config.padding[2]) {
                            pooling_step<scalar_t, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                        }
                        continue;
                    }

                    for (int in_x = out_x * STRIDE; in_x < out_x * STRIDE + KERNEL_SIZE; ++in_x){
                        const int in_x_im = in_x + tile_start_in_x;
                        if (in_x_im < 0) {
                            // pooling_step<scalar_t, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                            continue;
                        } 
                        if (in_x_im >= dim.in.w) {
                            // this is just to get same behaviour as when first applying padding and then pooling
                            if (in_x_im < dim.in.w + config.padding[3] && config.padding[1] != config.padding[3]) {
                                pooling_step<scalar_t, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                            }
                            continue;
                        }
                        
                        const bool valid = w_in > 8 ? 
                            (s_mask[(in_y * w_in + in_x)] != 0) :
                            ((t_mask & (1LLU << (in_y * w_in + in_x))) != 0);
                        
                        if (!valid && POOL_MODE == 1) {
                            // TODO improve like single pixel version
                            continue;
                        }

                        const int in_idx = (in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c;
                        const scalar_t delta = valid ? batch_in[in_idx] : scalar_t(0.0f);
                        const scalar_t prev_in = POOL_MODE == 0 ? batch_prev_in[in_idx] : 0.0f;
                        const scalar_t acc_val = delta + prev_in;

                        // if (blockIdx.x == 0 && threadIdx.x == 0 && out_y == 0 && out_x == 0) {
                        //     printf("in_y=%i in_x=%i in_y_im=%i in_x_im=%i tiy0=%i tix0=%i toy0=%i tox0=%i  valid=%i delta=%f prev_in=%f acc_val=%f\n",
                        //         in_y, in_x, in_y_im, in_x_im, tile_start_in_y, tile_start_in_x, tile_start_out_y, tile_start_out_x, 
                        //         (valid?1:0), delta, prev_in, acc_val
                        //     );
                        // }

                        pooling_step<scalar_t, POOL_MODE>(result, result_prev, acc_val, prev_in);
                    }
                }
                
                const int out_y_im = out_y * DILATION + tile_start_out_y;
                const int out_x_im = out_x * DILATION + tile_start_out_x;

                if (POOL_MODE == 0) {
                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = result - result_prev;
                }
                else {
                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = result / (config.kernel_size[0] * config.kernel_size[1]);
                }
            }
        }
    }
}




template<typename scalar_t = float, int KERNEL_SIZE=3, int pixelsPerBlock=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, int DILATION=1, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_less_registers_hp(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int32_t blocksPerBatch = divup(dim.out.w, pixelsPerBlock*DILATION) * divup(dim.out.h, pixelsPerBlock*DILATION)*DILATION*DILATION;
    const uint32_t batch = blockIdx.x / blocksPerBatch;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlock * pixelsPerBlock;
    const int K_HALF = KERNEL_SIZE / 2;
    const int K_HALF_RIGHT = (KERNEL_SIZE-1) / 2;

    const int tile_idx = blockIdx.x % blocksPerBatch;
    const int tiles_per_x = divup(dim.out.w, pixelsPerBlock * DILATION);
    const int dilation_idx = tile_idx / (DILATION * DILATION);
    const int dilation_sub_idx = tile_idx % (DILATION * DILATION);
    const int tile_start_out_y = (dilation_idx / tiles_per_x) * pixelsPerBlock * DILATION + (dilation_sub_idx / DILATION);
    const int tile_start_out_x = (dilation_idx % tiles_per_x) * pixelsPerBlock * DILATION + (dilation_sub_idx % DILATION);
    const int tile_start_in_y = tile_start_out_y * STRIDE - config.padding[0];
    const int tile_start_in_x = tile_start_out_x * STRIDE - config.padding[1];
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;
    const int w_in = pixelsPerBlock + (pixelsPerBlock-1) * (STRIDE-1) + (K_HALF+K_HALF_RIGHT);
    const int n_in_px = w_in * w_in;

    __shared__ uint32_t s_mask[n_in_px];

    for (int i = threadIdx.x; i < n_in_px; i += blockDim.x) {
        int y = tile_start_in_y + (i / w_in) * DILATION;
        int x = tile_start_in_x + (i % w_in) * DILATION;
        if (y >= 0 && y < dim.in.h && x >= 0 && x < dim.in.w) {
            const int mask_idx = y * dim.in.w + x;
            s_mask[i] = batch_mask != nullptr ? batch_mask[mask_idx] : 1;
        } else {
            s_mask[i] = 0;
        }
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
    //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
    //     );
    // }

    int density = 0;
    uint64_t t_mask = 0LLU;
    for (int i = 0; i < n_in_px; ++i) {
        if (s_mask[i] != 0) {
            t_mask += (1LLU << i);
            ++density;
        }
    }

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.z == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
        if (POOL_MODE == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * 2 * dim.in.c)); 
        } else {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * dim.in.c)); 
        }
    }
#endif

    if (out_mask != nullptr) {
        uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];
        for (int out_px = threadIdx.x; out_px < n_pixels_out; out_px += BLOCK_SIZE) {
            int p_y = out_px / pixelsPerBlock;
            int p_x = out_px % pixelsPerBlock;
            int out_y = p_y * DILATION + tile_start_out_y;
            int out_x = p_x * DILATION + tile_start_out_x; 
            if (out_y >= dim.out.h || out_x >= dim.out.w)
                continue;

            bool updated = false;
            for (int y = 0; y < KERNEL_SIZE; y++) {
                for (int x = 0; x < KERNEL_SIZE; x++) {
                    int in_px_idx = ((p_y*STRIDE) + y) * w_in + ((p_x*STRIDE) + x);
                    
                    updated |= w_in > 8 ? 
                        (s_mask[in_px_idx] != 0) :
                        ((t_mask & (1LLU << in_px_idx)) != 0);
                }
            }
            batch_out_mask[out_y * dim.out.w + out_x] = updated ? 1:0;
#ifdef ENABLE_METRICS
            if (updated) {
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
            }
#endif
        }
    }

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            for (int out_idx = threadIdx.x; out_idx < n_pixels_out * dim.out.c; out_idx += BLOCK_SIZE) {
                int out_px = out_idx / dim.out.c;
                int out_c = out_idx % dim.out.c;
                int p_y = out_px / pixelsPerBlock;
                int p_x = out_px % pixelsPerBlock;
                int out_y = p_y * DILATION + tile_start_out_y;
                int out_x = p_x * DILATION + tile_start_out_x; 
                if (out_y >= dim.out.h || out_x >= dim.out.w)
                    continue;
                batch_out[(out_y * dim.out.w + out_x) * dim.out.c + out_c] = __float2half(0.0f); 
            }
        }

        // if (threadIdx.x == 0) {
        //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
        //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
        //     );
        // }
        return;
    }

    // TODO check if half2 mode can be implemented

    for (int out_c = tile_start_z + threadIdx.x; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE) {
        for (int out_y = 0; out_y < pixelsPerBlock && out_y + tile_start_out_y < dim.out.h; ++out_y) {
            for (int out_x = 0; out_x < pixelsPerBlock && out_x + tile_start_out_x < dim.out.w; ++out_x) {
                float result = POOL_MODE == 0 ? -1e30f : 0.0f;
                float result_prev = POOL_MODE == 0 ? -1e30f : 0.0f;
                
                for (int in_y = out_y * STRIDE; in_y < out_y * STRIDE + KERNEL_SIZE; ++in_y){
                    const int in_y_im = in_y + tile_start_in_y;
                    if (in_y_im < 0) {
                        // pooling_step<scalar_t, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                        continue;
                    } 
                    if (in_y_im >= dim.in.h) {
                        // this is just to get same behaviour as when first applying padding and then pooling
                        if (in_y_im < dim.in.h + config.padding[2] && config.padding[0] != config.padding[2]) {
                            pooling_step<float, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                        }
                        continue;
                    }

                    for (int in_x = out_x * STRIDE; in_x < out_x * STRIDE + KERNEL_SIZE; ++in_x){
                        const int in_x_im = in_x + tile_start_in_x;
                        if (in_x_im < 0) {
                            // pooling_step<scalar_t, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                            continue;
                        } 
                        if (in_x_im >= dim.in.w) {
                            // this is just to get same behaviour as when first applying padding and then pooling
                            if (in_x_im < dim.in.w + config.padding[3] && config.padding[1] != config.padding[3]) {
                                pooling_step<float, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                            }
                            continue;
                        }
                        
                        const bool valid = w_in > 8 ? 
                            (s_mask[(in_y * w_in + in_x)] != 0) :
                            ((t_mask & (1LLU << (in_y * w_in + in_x))) != 0);
                        
                        if (!valid && POOL_MODE == 1) {
                            // TODO improve like single pixel version
                            continue;
                        }

                        const int in_idx = (in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c;
                        const float delta = valid ? __half2float(batch_in[in_idx]) : 0.0f;
                        const float prev_in = POOL_MODE == 0 ? __half2float(batch_prev_in[in_idx]) : 0.0f;
                        const float acc_val = delta + prev_in;

                        // if (blockIdx.x == 0 && threadIdx.x == 0 && out_y == 0 && out_x == 0) {
                        //     printf("in_y=%i in_x=%i in_y_im=%i in_x_im=%i tiy0=%i tix0=%i toy0=%i tox0=%i  valid=%i delta=%f prev_in=%f acc_val=%f\n",
                        //         in_y, in_x, in_y_im, in_x_im, tile_start_in_y, tile_start_in_x, tile_start_out_y, tile_start_out_x, 
                        //         (valid?1:0), delta, prev_in, acc_val
                        //     );
                        // }

                        pooling_step<float, POOL_MODE>(result, result_prev, acc_val, prev_in);
                    }
                }
                
                const int out_y_im = out_y * DILATION + tile_start_out_y;
                const int out_x_im = out_x * DILATION + tile_start_out_x;

                if (POOL_MODE == 0) {
                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __float2half(result - result_prev);
                }
                else {
                    // batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __hdiv(result, __int2half_rd(config.kernel_size[0] * config.kernel_size[1]));
                    // dividing here with large kernel sizes can be problematic due to the limited resolution of half --> some kernels are 256x256 = 65k = half::max
                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __float2half(result / (config.kernel_size[0] * config.kernel_size[1]));
                }
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_single_element(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int pixelsPerBlock = 1;
    const int32_t blocksPerBatch = divup(dim.out.w, pixelsPerBlock*config.dilation[1]) * divup(dim.out.h, pixelsPerBlock*config.dilation[0])*config.dilation[0]*config.dilation[1];
    const uint32_t batch = blockIdx.x / blocksPerBatch;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask + (batch * dim.in.h * dim.in.w);

    const int tile_idx = blockIdx.x % blocksPerBatch;
    const int tiles_per_x = divup(dim.out.w, pixelsPerBlock * config.dilation[1]);
    const int dilation_idx = tile_idx / (config.dilation[0] * config.dilation[0]);
    const int dilation_sub_idx = tile_idx % (config.dilation[0] * config.dilation[1]);
    const int tile_start_out_y = (dilation_idx / tiles_per_x) * pixelsPerBlock * config.dilation[1] + (dilation_sub_idx / config.dilation[1]);
    const int tile_start_out_x = (dilation_idx % tiles_per_x) * pixelsPerBlock * config.dilation[1] + (dilation_sub_idx % config.dilation[1]);
    const int tile_start_in_y = tile_start_out_y * config.stride[0] - config.padding[0];
    const int tile_start_in_x = tile_start_out_x * config.stride[1] - config.padding[1];
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;


    if (tile_start_out_y >= dim.out.h || tile_start_out_x >= dim.out.w) {
        // can happen in dilated mode
        return;
    }
#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.c * dim.in.w * dim.in.h)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.in.c * dim.out.w * dim.out.h)); 
    }
#endif

    for (int out_c = tile_start_z + threadIdx.x; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE) {
        scalar_t result = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
        scalar_t result_prev = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
        bool any_active = false;
        
        for (int in_y = 0; in_y < config.kernel_size[0]*config.dilation[0]; in_y += config.dilation[0]){
            const int in_y_im = in_y + tile_start_in_y;
            if (in_y_im < 0 || in_y_im >= dim.in.h)
                continue;

            const int y_in_off = in_y_im*dim.in.w;

            for (int in_x = 0; in_x < config.kernel_size[1]*config.dilation[1]; in_x += config.dilation[1]){
                const int in_x_im = in_x + tile_start_in_x;
                if (in_x_im < 0 || in_x_im >= dim.in.w)
                    continue;
                    
                const int in_idx_2d = y_in_off + in_x_im;
                const bool valid = mask == nullptr || batch_mask[in_idx_2d] != 0;
#ifdef ENABLE_METRICS
                if (valid && threadIdx.x == 0) {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                }
#endif
                any_active |= valid;
                if (!valid && POOL_MODE == 1) {
                    // TODO improve like single pixel version
                    continue;
                }
                
                const int in_idx = in_idx_2d * dim.in.c + out_c;

                const scalar_t delta = valid ? batch_in[in_idx] : scalar_t(0.0f);
                const scalar_t prev_in = POOL_MODE == 0 ? batch_prev_in[in_idx] : 0.0f;
                const scalar_t acc_val = delta + prev_in;

                // if (threadIdx.x%32 == 0 && tile_start_out_y == 1 && tile_start_out_x == 0) {
                //     printf("in_y=%i in_x=%i in_y_im=%i in_x_im=%i tiy0=%i tix0=%i toy0=%i tox0=%i valid=%i delta=%i prev_in=%i acc_val=%i\n",
                //         in_y, in_x, in_y_im, in_x_im, tile_start_in_y, tile_start_in_x, tile_start_out_y, tile_start_out_x, 
                //         (valid?1:0), delta, prev_in, acc_val
                //     );
                // }

                pooling_step<scalar_t, POOL_MODE>(result, result_prev, acc_val, prev_in);
            }
        }
        if (out_c == 0 && out_mask != nullptr) {
            uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];
            batch_out_mask[tile_start_out_y * dim.out.w + tile_start_out_x] = any_active ? 1:0;
        }
        if (!any_active)
            return;

#ifdef ENABLE_METRICS
        if (threadIdx.x == 0) {
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
        }
#endif
                
        const int out_y_im = tile_start_out_y;
        const int out_x_im = tile_start_out_x;

        if (POOL_MODE == 0) {
            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = result - result_prev;
        }
        else {
            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = result / (config.kernel_size[0] * config.kernel_size[1]);
        }
    }
}




template<typename scalar_t = float, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_single_element_hp(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int pixelsPerBlock = 1;
    const int32_t blocksPerBatch = divup(dim.out.w, pixelsPerBlock*config.dilation[1]) * divup(dim.out.h, pixelsPerBlock*config.dilation[0])*config.dilation[0]*config.dilation[1];
    const uint32_t batch = blockIdx.x / blocksPerBatch;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask + (batch * dim.in.h * dim.in.w);

    const int tile_idx = blockIdx.x % blocksPerBatch;
    const int tiles_per_x = divup(dim.out.w, pixelsPerBlock * config.dilation[1]);
    const int dilation_idx = tile_idx / (config.dilation[0] * config.dilation[0]);
    const int dilation_sub_idx = tile_idx % (config.dilation[0] * config.dilation[1]);
    const int tile_start_out_y = (dilation_idx / tiles_per_x) * pixelsPerBlock * config.dilation[1] + (dilation_sub_idx / config.dilation[1]);
    const int tile_start_out_x = (dilation_idx % tiles_per_x) * pixelsPerBlock * config.dilation[1] + (dilation_sub_idx % config.dilation[1]);
    const int tile_start_in_y = tile_start_out_y * config.stride[0] - config.padding[0];
    const int tile_start_in_x = tile_start_out_x * config.stride[1] - config.padding[1];
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;


    if (tile_start_out_y >= dim.out.h || tile_start_out_x >= dim.out.w) {
        // can happen in dilated mode
        return;
    }

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(dim.in.w * dim.in.h * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.out.w * dim.out.h * dim.out.c)); 
    }
#endif


    // TODO check if half2 mode can be implemented

    for (int out_c = tile_start_z + threadIdx.x; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE) {
        float result = POOL_MODE == 0 ? -1e30f : 0.0f;
        float result_prev = POOL_MODE == 0 ? -1e30f : 0.0f;
        bool any_active = false;
        
        for (int in_y = 0; in_y < config.kernel_size[0]*config.dilation[0]; in_y += config.dilation[0]){
            const int in_y_im = in_y + tile_start_in_y;
            if (in_y_im < 0 || in_y_im >= dim.in.h) {
                if (in_y_im >= dim.in.h && POOL_MODE == 0) {
                    pooling_step<float, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                }
                continue;
            }

            const int y_in_off = in_y_im*dim.in.w;

            for (int in_x = 0; in_x < config.kernel_size[1]*config.dilation[1]; in_x += config.dilation[1]){
                const int in_x_im = in_x + tile_start_in_x;
                if (in_x_im < 0 || in_x_im >= dim.in.w) {
                    if (in_x_im >= dim.in.w && POOL_MODE == 0) {
                        pooling_step<float, POOL_MODE>(result, result_prev, 0.0f, 0.0f);
                    }
                    continue;
                }
                
                const int in_idx_2d = y_in_off + in_x_im;
                const bool valid = mask == nullptr || batch_mask[in_idx_2d] != 0;
                any_active |= valid;
                if (!valid && POOL_MODE == 1) {
                    // TODO improve like single pixel version
                    continue;
                }
#ifdef ENABLE_METRICS
                if (valid && threadIdx.x == 0) {
                    if (POOL_MODE == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    } else {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                    }
                }
#endif
                const int in_idx = in_idx_2d * dim.in.c + out_c;

                const float delta = valid ? __half2float(batch_in[in_idx]) : 0.0f;
                const float prev_in = POOL_MODE == 0 ? __half2float(batch_prev_in[in_idx]) : 0.0f;
                const float acc_val = delta + prev_in;

                // if (threadIdx.x%32 == 0 && tile_start_out_y == 1 && tile_start_out_x == 0) {
                //     printf("in_y=%i in_x=%i in_y_im=%i in_x_im=%i tiy0=%i tix0=%i toy0=%i tox0=%i valid=%i delta=%i prev_in=%i acc_val=%i\n",
                //         in_y, in_x, in_y_im, in_x_im, tile_start_in_y, tile_start_in_x, tile_start_out_y, tile_start_out_x, 
                //         (valid?1:0), delta, prev_in, acc_val
                //     );
                // }

                pooling_step<float, POOL_MODE>(result, result_prev, acc_val, prev_in);
            }
        }
        if (out_c == 0 && out_mask != nullptr) {
            uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];
            batch_out_mask[tile_start_out_y * dim.out.w + tile_start_out_x] = any_active ? 1:0;
        }
        if (!any_active)
            return;

#ifdef ENABLE_METRICS
        if (any_active && threadIdx.x == 0) {
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
        }
#endif
                
        const int out_y_im = tile_start_out_y;
        const int out_x_im = tile_start_out_x;

        if (POOL_MODE == 0) {
            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __float2half(result - result_prev);
        }
        else {
            // batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __hdiv(result, __int2half_rd(config.kernel_size[0] * config.kernel_size[1]));
            // dividing here with large kernel sizes can be problematic due to the limited resolution of half --> some kernels are 256x256 = 65k = half::max
            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __float2half(result / (config.kernel_size[0] * config.kernel_size[1]));
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=32, int OUT_CHANNELS_PER_BLOCK=4, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_warp_per_channel(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int pixelsPerBlock = 1;
    const int32_t blocksPerBatch = divup(dim.out.w, pixelsPerBlock*config.dilation[1]) * divup(dim.out.h, pixelsPerBlock*config.dilation[0])*config.dilation[0]*config.dilation[1];
    const uint32_t batch = blockIdx.x / blocksPerBatch;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask + (batch * dim.in.h * dim.in.w);

    const int tile_idx = blockIdx.x % blocksPerBatch;
    const int tiles_per_x = divup(dim.out.w, pixelsPerBlock * config.dilation[1]);
    const int dilation_idx = tile_idx / (config.dilation[0] * config.dilation[0]);
    const int dilation_sub_idx = tile_idx % (config.dilation[0] * config.dilation[1]);
    const int tile_start_out_y = (dilation_idx / tiles_per_x) * pixelsPerBlock * config.dilation[1] + (dilation_sub_idx / config.dilation[1]);
    const int tile_start_out_x = (dilation_idx % tiles_per_x) * pixelsPerBlock * config.dilation[1] + (dilation_sub_idx % config.dilation[1]);
    const int tile_start_in_y = tile_start_out_y * config.stride[0] - config.padding[0];
    const int tile_start_in_x = tile_start_out_x * config.stride[1] - config.padding[1];
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;
    const int w_in = pixelsPerBlock + (pixelsPerBlock-1) * (config.stride[0]-1) + (config.kernel_size[0]-1);


    if (tile_start_out_y >= dim.out.h || tile_start_out_x >= dim.out.w) {
        // can happen in dilated mode
        return;
    }

    const int n_in_px = config.kernel_size[0] * config.kernel_size[1];     
    const int n_pixels_out = pixelsPerBlock*pixelsPerBlock;     

    __shared__ int density;
    if (threadIdx.x == 0) {
        density = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_in_px; i += blockDim.x) {
        int y = tile_start_in_y + (i / w_in) * config.dilation[0];
        int x = tile_start_in_x + (i % w_in) * config.dilation[1];
        int active = 0;
        if (y >= 0 && y < dim.in.h && x >= 0 && x < dim.in.w) {
            const int mask_idx = y * dim.in.w + x;
            active = batch_mask != nullptr ? batch_mask[mask_idx] : 1;
        } else {
            active = 0;
        }
        if (active != 0) {
            atomicAdd(&density, 1);
        }
    }
    __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("tile_idx=%i tiles_per_x=%i dilation_idx=%i dilation_sub_idx=%i tile_start_out_y=%i tile_start_out_x=%i tile_start_in_y=%i tile_start_in_x=%i, tile_start_z=%i w_in=%i density=%i\n",
    //     tile_idx, tiles_per_x, dilation_idx, dilation_sub_idx, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, w_in, density
    //     );
    // }

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.z == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
        if (POOL_MODE == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * 2 * dim.in.c)); 
        } else {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(density * dim.in.c)); 
        }
    }
#endif

    // if (threadIdx.x == 0) {
    //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
    //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
    //     );
    // }

    if (out_mask != nullptr && tile_start_z == 0) {
        uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];
        for (int out_px = threadIdx.x; out_px < n_pixels_out; out_px += BLOCK_SIZE) {
            int p_y = out_px / pixelsPerBlock;
            int p_x = out_px % pixelsPerBlock;
            int out_y = p_y * config.dilation[0] + tile_start_out_y;
            int out_x = p_x * config.dilation[1] + tile_start_out_x; 
            if (out_y >= dim.out.h || out_x >= dim.out.w)
                continue;

            bool updated = density != 0;
            batch_out_mask[out_y * dim.out.w + out_x] = updated ? 1:0;
        }
    }

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            for (int out_idx = threadIdx.x; out_idx < n_pixels_out * dim.out.c; out_idx += BLOCK_SIZE) {
                int out_px = out_idx / dim.out.c;
                int out_c = out_idx % dim.out.c;
                int p_y = out_px / pixelsPerBlock;
                int p_x = out_px % pixelsPerBlock;
                int out_y = p_y * config.dilation[0] + tile_start_out_y;
                int out_x = p_x * config.dilation[1] + tile_start_out_x; 
                if (out_y >= dim.out.h || out_x >= dim.out.w)
                    continue;
                batch_out[(out_y * dim.out.w + out_x) * dim.out.c + out_c] = 0.0f; 
#ifdef ENABLE_METRICS
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
#endif
            }
        }

        // if (threadIdx.x == 0) {
        //     printf("bId=%i, out_y=%i out_x=%i, in_y=%i in_x=%i stride=%i w_in=%i\n",
        //         blockIdx.x, tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, STRIDE, w_in
        //     );
        // }
        return;
    }

    scalar_t result[OUT_CHANNELS_PER_BLOCK];
    scalar_t result_prev[OUT_CHANNELS_PER_BLOCK];

    #pragma unroll
    for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
        result[out_c] = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
        result_prev[out_c] = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
    }


    for (int px_idx = threadIdx.x; px_idx < n_in_px; px_idx += WARP_SIZE) {
        const int in_y = px_idx / config.kernel_size[0];
        const int in_x = px_idx % config.kernel_size[1];
        const int in_y_im = in_y * config.dilation[0] + tile_start_in_y;
        const int in_x_im = in_x * config.dilation[1] + tile_start_in_x;
        const int in_idx = (in_y_im * dim.in.w + in_x_im) * dim.in.c + tile_start_z;
        const bool inside = in_y_im >= 0 && in_y_im < dim.in.h && in_x_im >= 0 && in_x_im < dim.in.w;
        const bool valid = inside && (mask == nullptr || batch_mask[in_y_im * dim.in.w + in_x_im] != 0);
        
        // printf("tId=%i in_y=%i in_x=%i in_y_im=%i in_x_im=%i,in_idx=%i, valid=%i\n",
        //     threadIdx.x, in_y, in_x, in_y_im, in_x_im, in_idx, (valid?1:0)
        // );

        if (!valid && POOL_MODE == 1) {
            // TODO improve like single pixel version
            continue;
        }

        scalar_t delta[OUT_CHANNELS_PER_BLOCK];
        if (valid) {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
                if (out_c + tile_start_z < dim.out.c) {
                    delta[out_c] = batch_in[in_idx + out_c];
                }
            }
        } else {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
                delta[out_c] = scalar_t(0.0f);
            }
        }

        #pragma unroll
        for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
            if (out_c + tile_start_z < dim.out.c) {
                const scalar_t prev_in = POOL_MODE == 0 && inside ? batch_prev_in[in_idx + out_c] : 0.0f;
                const scalar_t acc_val = delta[out_c] + prev_in;
                pooling_step<scalar_t, POOL_MODE>(result[out_c], result_prev[out_c], acc_val, prev_in);

                // if (blockIdx.x == 0 && blockIdx.z == 0) {
                //     printf("in_y=%i in_x=%i in_y_im=%i in_x_im=%i in_idx=%i dim.in.c=%i out_c=%i valid=%i delta=%f prev_in=%f tId=%i\n", 
                //         in_y, in_x, in_y_im, in_x_im, in_idx, dim.in.c, out_c, (valid?1:0), delta[out_c], prev_in, threadIdx.x
                //     );
                // } 
            }
        }

    }

    if (POOL_MODE == 0) {
        for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
            // if (blockIdx.x == 0 && blockIdx.z == 0) {
            //     printf("before: out_c=%i, result=%f, result_prev=%f tId=%i\n", out_c, result[out_c], result_prev[out_c], threadIdx.x);
            // } 
            result[out_c] = Utils::warpReduceMax(result[out_c]);
            result_prev[out_c] = Utils::warpReduceMax(result_prev[out_c]);

            // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.z == 0) {
            //     printf("after: out_c=%i, result=%f, result_prev=%f\n", out_c, result[out_c], result_prev[out_c]);
            // } 
        }
    } else {
        for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
            result[out_c] = Utils::warpReduceSum(result[out_c]);
        }
    }

    if (threadIdx.x == 0) {
        if (POOL_MODE == 0) {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
                if (out_c + tile_start_z < dim.out.c) {
                    batch_out[(tile_start_out_y * dim.out.w + tile_start_out_x) * dim.out.c + tile_start_z + out_c] = result[out_c] - result_prev[out_c];
                }
            }
        }
        else {
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_BLOCK; ++out_c) {
                if (out_c + tile_start_z < dim.out.c) {
                    batch_out[(tile_start_out_y * dim.out.w + tile_start_out_x) * dim.out.c + tile_start_z + out_c] = result[out_c] / n_in_px;
                }
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=32, int pixelsPerBlock=1, int OUT_CHANNELS_PER_THREAD=4, int OUT_CHANNELS_PER_BLOCK=128, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px(
    const scalar_t* input,
    const scalar_t* prev_input,
    scalar_t* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const uint32_t batch = blockIdx.y;
    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : (mask + (batch * dim.in.h * dim.in.w));

    const int n_in_px = dim.in.w * dim.in.h;
    const int start_idx = blockIdx.x * pixelsPerBlock;
    const int end_idx = min(start_idx + pixelsPerBlock, n_in_px);
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.z == 0) {
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.out.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
        }
    }
#endif

    bool any_active = false;

    __shared__ uint32_t s_mask[pixelsPerBlock];

    for (int i = threadIdx.x; start_idx + i < end_idx; i += BLOCK_SIZE) {
        s_mask[i] = batch_mask == nullptr ? 1 : batch_mask[i + start_idx];
    }

    __syncthreads();

    for (int i = 0; start_idx + i < end_idx; ++i) {
        any_active |= s_mask[i] != 0;
    }

    if (!any_active) {
        return;
    }

    if (tile_start_z == 0 && threadIdx.x == 0 && out_mask != nullptr) {
        out_mask[batch] = 1;
    }


    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || (out_c_off-tile_start_z) < OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE * OUT_CHANNELS_PER_THREAD) {
        int t_out_c = out_c_off + threadIdx.x * OUT_CHANNELS_PER_THREAD;
        scalar_t result[OUT_CHANNELS_PER_THREAD];
        scalar_t result_prev[OUT_CHANNELS_PER_THREAD];

        #pragma unroll
        for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
            result[out_c] = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
            result_prev[out_c] = POOL_MODE == 0 ? scalar_t(-1e30f) : scalar_t(0.0f);
        }

        if (blockIdx.x == 0) {
            // TODO: check if this also applies to padding[0] and padding[1]
            if ((config.padding[2] > 0 || config.padding[3] > 0) && POOL_MODE == 0) {
                for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                    pooling_step<float, POOL_MODE>(result[out_c], result_prev[out_c], 0.0f, 0.0f);
                }
            }
        }


        for (int in_idx = start_idx; in_idx < end_idx; ++in_idx) {
            const bool valid = mask == nullptr || s_mask[in_idx-start_idx] != 0;
            const int in_idx_im = in_idx * dim.out.c;

            scalar_t delta[OUT_CHANNELS_PER_THREAD];
            if (valid) {
                #pragma unroll
                for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                    if (out_c + t_out_c < dim.out.c) {
                        delta[out_c] = batch_in[in_idx_im + t_out_c + out_c];
                    }
                }
#ifdef ENABLE_METRICS
                if (threadIdx.x == 0) {
                    if (POOL_MODE == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    } else {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                    }
                }
#endif
            } else {
                if (POOL_MODE == 1) {
                    continue;
                }
                else {
                    #pragma unroll
                    for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                        delta[out_c] = scalar_t(0.0f);
                    }
                }
            }

            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                if (out_c + t_out_c < dim.out.c) {
                    const scalar_t prev_in = POOL_MODE == 0 ? batch_prev_in[in_idx_im + t_out_c + out_c] : 0.0f;
                    const scalar_t acc_val = delta[out_c] + prev_in;
                    pooling_step<scalar_t, POOL_MODE>(result[out_c], result_prev[out_c], acc_val, prev_in);

                    // if (threadIdx.x == 0 && blockIdx.z == 0) {
                    //     int in_y = in_idx / dim.in.w;
                    //     int in_x = in_idx % dim.in.w;
                    //     printf("in_y=%i in_x=%i in_idx=%i dim.in.c=%i out_c=%i valid=%i delta=%f prev_in=%f bId=%i\n", 
                    //         in_y, in_x, in_idx, dim.in.c, out_c, (valid?1:0), delta[out_c], prev_in, blockIdx.x
                    //     );
                    // } 
                }
            }

        }

        if (POOL_MODE == 0) {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                // if (threadIdx.x == 0) {
                //     printf("result=%f, result_prev=%f out_c=%i, bIdx=%i bIdz=%i\n", result[out_c], result_prev[out_c], out_c, blockIdx.x, blockIdx.z);
                // } 
                if (out_c + t_out_c < dim.out.c) {
                    Utils::atomicMax(&batch_out[t_out_c + out_c], result[out_c] - result_prev[out_c]);
                }
            }
        }
        else {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                // if (threadIdx.x == 0) {
                //     printf("result=%f, result_prev=%f out_c=%i, bIdx=%i bIdz=%i tIdx=%i\n", result[out_c], result_prev[out_c], out_c, blockIdx.x, blockIdx.z, threadIdx.x);
                // } 
                if (out_c + t_out_c < dim.out.c) {
                    atomicAdd(&batch_out[t_out_c + out_c], result[out_c] / n_in_px);
                }
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=32, int pixelsPerBlock=1, int OUT_CHANNELS_PER_THREAD=4, int OUT_CHANNELS_PER_BLOCK=128, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int POOL_MODE=0>
__global__ void deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px_hp(
    const scalar_t* input,
    const scalar_t* prev_input,
    float* output,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const uint32_t batch = blockIdx.y;
    float* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const scalar_t* batch_prev_in = prev_input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : (mask + (batch * dim.in.h * dim.in.w));

    const int n_in_px = dim.in.w * dim.in.h;
    const int start_idx = blockIdx.x * pixelsPerBlock;
    const int end_idx = min(start_idx + pixelsPerBlock, n_in_px);
    const int tile_start_z = FULL_DEPTH ? 0 : blockIdx.z * OUT_CHANNELS_PER_BLOCK;

#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.z == 0) {
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(dim.out.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
        }
    }
#endif

    bool any_active = false;

    __shared__ uint32_t s_mask[pixelsPerBlock];

    for (int i = threadIdx.x; start_idx + i < end_idx; i += BLOCK_SIZE) {
        s_mask[i] = batch_mask == nullptr ? 1 : batch_mask[i + start_idx];
    }

    __syncthreads();

    for (int i = 0; start_idx + i < end_idx; ++i) {
        any_active |= s_mask[i] != 0;
    }

    if (!any_active) {
        return;
    }

    if (tile_start_z == 0 && threadIdx.x == 0 && out_mask != nullptr) {
        out_mask[batch] = 1;
    }


    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || (out_c_off-tile_start_z) < OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE * OUT_CHANNELS_PER_THREAD) {
        int t_out_c = out_c_off + threadIdx.x * OUT_CHANNELS_PER_THREAD;
        float result[OUT_CHANNELS_PER_THREAD];
        float result_prev[OUT_CHANNELS_PER_THREAD];

        #pragma unroll
        for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
            result[out_c] = POOL_MODE == 0 ? -1e30f : 0.0f;
            result_prev[out_c] = POOL_MODE == 0 ? -1e30f : 0.0f;
        }

        if (blockIdx.x == 0) {
            // TODO: check if this also applies to padding[0] and padding[1]
            if ((config.padding[2] > 0 || config.padding[3] > 0) && POOL_MODE == 0) {
                for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                    pooling_step<float, POOL_MODE>(result[out_c], result_prev[out_c], 0.0f, 0.0f);
                }
            }
        }


        for (int in_idx = start_idx; in_idx < end_idx; ++in_idx) {
            const bool valid = mask == nullptr || s_mask[in_idx-start_idx] != 0;
            const int in_idx_im = in_idx * dim.out.c;

            float delta[OUT_CHANNELS_PER_THREAD];
            if (valid) {
                #pragma unroll
                for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                    if (out_c + t_out_c < dim.out.c) {
                        delta[out_c] = __half2float(batch_in[in_idx_im + t_out_c + out_c]);
                    }
                }
#ifdef ENABLE_METRICS
                if (threadIdx.x == 0) {
                    if (POOL_MODE == 0) {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(2 * dim.in.c)); 
                    } else {
                        atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
                    }
                }
#endif
            } else {
                if (POOL_MODE == 1) {
                    continue;
                }
                else {
                    #pragma unroll
                    for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                        delta[out_c] = 0.0f;
                    }
                }
            }

            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                if (out_c + t_out_c < dim.out.c) {
                    const float prev_in = POOL_MODE == 0 ? __half2float(batch_prev_in[in_idx_im + t_out_c + out_c]) : 0.0f;
                    const float acc_val = delta[out_c] + prev_in;
                    pooling_step<float, POOL_MODE>(result[out_c], result_prev[out_c], acc_val, prev_in);

                    // if (threadIdx.x == 0 && blockIdx.z == 0) {
                    //     int in_y = in_idx / dim.in.w;
                    //     int in_x = in_idx % dim.in.w;
                    //     printf("in_y=%i in_x=%i in_idx=%i dim.in.c=%i out_c=%i valid=%i delta=%f prev_in=%f bId=%i\n", 
                    //         in_y, in_x, in_idx, dim.in.c, out_c, (valid?1:0), delta[out_c], prev_in, blockIdx.x
                    //     );
                    // } 
                }
            }

        }

        if (POOL_MODE == 0) {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                // if (threadIdx.x == 0) {
                //     printf("result=%f, result_prev=%f out_c=%i, bIdx=%i bIdz=%i\n", result[out_c], result_prev[out_c], out_c, blockIdx.x, blockIdx.z);
                // } 
                // if (threadIdx.x == 0 || true) {
                //     printf("result=%f, result_prev=%f t_out_c=%i out_c=%i, bIdx=%i bIdz=%i tIdx=%i\n", result[out_c], result_prev[out_c], t_out_c, out_c, blockIdx.x, blockIdx.z, threadIdx.x);
                // } 
                if (out_c + t_out_c < dim.out.c) {
                    Utils::atomicMax(&batch_out[t_out_c + out_c], result[out_c] - result_prev[out_c]);
                }
            }
        }
        else {
            #pragma unroll
            for (int out_c = 0; out_c < OUT_CHANNELS_PER_THREAD; ++out_c) {
                // if (threadIdx.x == 0 || true) {
                //     printf("result=%f, result_prev=%f t_out_c=%i out_c=%i, bIdx=%i bIdz=%i tIdx=%i batch=%i out=%p, batch_out=%p\n", 
                //         result[out_c], result_prev[out_c], t_out_c, out_c, blockIdx.x, blockIdx.z, threadIdx.x, batch, (void*)output, (void*)batch_out
                //     );
                // } 
                if (out_c + t_out_c < dim.out.c) {
                    atomicAdd(&batch_out[t_out_c + out_c], result[out_c] / n_in_px);
                }
            }
        }
    }
}



template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, int SCALE=2>
__global__ void deltacnn_sparse_upsample_kernel(const scalar_t * __restrict__ val_in, scalar_t * __restrict__ val_out, const uint32_t *mask_in, uint32_t *mask_out, Dimensions dim) {
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t((end_idx - start_idx) * dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx - start_idx) * SCALE * SCALE * dim.in.c)); 
        }
#endif

    if (!checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask_in)) {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            int in_y = px_idx / dim.in.w;
            int in_x = px_idx % dim.in.w;
            int out_y_off = in_y * SCALE;
            int out_x_off = in_x * SCALE;
            for (int i = lane_idx; i < SCALE * SCALE; i += WARP_SIZE) {
                int y = i / SCALE;
                int x = i % SCALE;
                int y_off = (out_y_off + y) * dim.out.w + out_x_off;
                int out_px_idx = y_off + x;
                mask_out[out_px_idx] = 0;
            }
        }

        return;
    }

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
        int in_y = px_idx / dim.in.w;
        int in_x = px_idx % dim.in.w;
        int out_y_off = in_y * SCALE;
        int out_x_off = in_x * SCALE;

        bool active = mask_in[px_idx] != 0;
        for (int i = lane_idx; i < SCALE * SCALE; i += WARP_SIZE) {
            int y = i / SCALE;
            int x = i % SCALE;
            int y_off = (out_y_off + y) * dim.out.w + out_x_off;
            int out_px_idx = y_off + x;
            mask_out[out_px_idx] = active ? 1: 0;
        }

        if (!active) {
            continue;
        }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(SCALE * SCALE * dim.in.c)); 
        }
#endif

        const scalar_t *in_px = &val_in[px_idx * dim.in.c];
        for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
            scalar_t val = in_px[c];

            #pragma unroll
            for (int y = 0; y < SCALE; ++y) {
                int y_off = (out_y_off + y) * dim.out.w + out_x_off;
                #pragma unroll
                for (int x = 0; x < SCALE; ++x) {
                    int out_px_idx = y_off + x;
                    val_out[out_px_idx * dim.out.c + c] = val;
                }
            }
        }
    }
}


template<typename scalar_t = half, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32, int SCALE=2>
__global__ void deltacnn_sparse_upsample_hp_kernel(const scalar_t * __restrict__ val_in, scalar_t * __restrict__ val_out, const uint32_t *mask_in, uint32_t *mask_out, Dimensions dim) {
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t((end_idx - start_idx) * dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx - start_idx) * SCALE * SCALE * dim.in.c)); 
        }
#endif

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
        int in_y = px_idx / dim.in.w;
        int in_x = px_idx % dim.in.w;
        int out_y_off = in_y * SCALE;
        int out_x_off = in_x * SCALE;

        bool active = mask_in[px_idx] != 0;
        for (int i = lane_idx; i < SCALE * SCALE; i += WARP_SIZE) {
            int y = i / SCALE;
            int x = i % SCALE;
            int y_off = (out_y_off + y) * dim.out.w + out_x_off;
            int out_px_idx = y_off + x;
            mask_out[out_px_idx] = active ? 1: 0;
        }

        if (!active) {
            continue;
        }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(SCALE * SCALE * dim.in.c)); 
        }
#endif

        if (dim.in.c % 2 == 0) {
            const scalar_t *in_px = &val_in[px_idx * dim.in.c];
            for (int c = lane_idx * 2; c < dim.in.c; c += WARP_SIZE * 2) {
                half2 val = *reinterpret_cast<const half2*>(&in_px[c]);

                #pragma unroll
                for (int y = 0; y < SCALE; ++y) {
                    int y_off = (out_y_off + y) * dim.out.w + out_x_off;
                    #pragma unroll
                    for (int x = 0; x < SCALE; ++x) {
                        int out_px_idx = y_off + x;
                        *reinterpret_cast<half2*>(&val_out[out_px_idx * dim.out.c + c]) = val;
                    }
                }
            }
        } else {
            const scalar_t *in_px = &val_in[px_idx * dim.in.c];
            for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                scalar_t val = in_px[c];

                #pragma unroll
                for (int y = 0; y < SCALE; ++y) {
                    int y_off = (out_y_off + y) * dim.out.w + out_x_off;
                    #pragma unroll
                    for (int x = 0; x < SCALE; ++x) {
                        int out_px_idx = y_off + x;
                        val_out[out_px_idx * dim.out.c + c] = val;
                    }
                }
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32>
__global__ void deltacnn_sparse_concatenate_kernel(
        const scalar_t * __restrict__ val_a, 
        const scalar_t * __restrict__ val_b, 
        scalar_t * __restrict__ val_out, 
        uint32_t *mask_a, 
        uint32_t *mask_b, 
        uint32_t *mask_out,
        Dimensions dim
        ) 
{
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
    if (lane_idx == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t((end_idx - start_idx) * dim.out.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx - start_idx) * dim.out.c)); 
    }
#endif

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
        bool active_a = mask_a[px_idx] != 0;
        bool active_b = mask_b[px_idx] != 0;

        if (!(active_a || active_b)) {
            if (lane_idx == 0) {
                mask_out[px_idx] = 0;
            }
            continue;
        }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            int active_channels = active_a ? (active_b ? dim.out.c : dim.in.c) : (dim.out.c-dim.in.c);
            atomicAdd(&d_metrics->n_vals_read, uint64_t(active_channels)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
        }
#endif
        
        if (lane_idx == 0) {
            mask_out[px_idx] = 1;
        }


        const scalar_t *a_px = &val_a[px_idx * dim.in.c];
        const scalar_t *b_px = &val_b[px_idx * (dim.out.c - dim.in.c)];
        scalar_t *out_px = &val_out[px_idx * dim.out.c];

        if (active_a) {
            for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                out_px[c] = a_px[c];
            }
        } else {
            for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                out_px[c] = scalar_t(0.0f);
            }
        }

        if (active_b) {
            for (int c = lane_idx; c < dim.out.c - dim.in.c; c += WARP_SIZE) {
                out_px[c + dim.in.c] = b_px[c];
            }
        } else {
            for (int c = lane_idx; c < dim.out.c - dim.in.c; c += WARP_SIZE) {
                out_px[c + dim.in.c] = scalar_t(0.0f);
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE=128, int PIXELS_PER_BLOCK = 32>
__global__ void deltacnn_sparse_concatenate_kernel_hp(
        const scalar_t * __restrict__ val_a, 
        const scalar_t * __restrict__ val_b, 
        scalar_t * __restrict__ val_out, 
        uint32_t *mask_a, 
        uint32_t *mask_b, 
        uint32_t *mask_out,
        Dimensions dim
        ) 
{
    const int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);

#ifdef ENABLE_METRICS
    if (lane_idx == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t((end_idx - start_idx) * dim.out.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx - start_idx) * dim.out.c)); 
    }
#endif

    for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
        bool active_a = mask_a[px_idx] != 0;
        bool active_b = mask_b[px_idx] != 0;

        if (!(active_a || active_b)) {
            if (lane_idx == 0) {
                mask_out[px_idx] = 0;
            }
            continue;
        }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            int active_channels = active_a ? (active_b ? dim.out.c : dim.in.c) : (dim.out.c-dim.in.c);
            atomicAdd(&d_metrics->n_vals_read, uint64_t(active_channels)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
        }
#endif
        
        if (lane_idx == 0) {
            mask_out[px_idx] = 1;
        }

        if (dim.in.c % 2 == 0 && dim.out.c % 2 == 0) {
            const half2 *a_px = reinterpret_cast<const half2*>(&val_a[px_idx * dim.in.c]);
            const half2 *b_px = reinterpret_cast<const half2*>(&val_b[px_idx * (dim.out.c - dim.in.c)]);
            half2 *out_px = reinterpret_cast<half2*>(&val_out[px_idx * dim.out.c]);

            if (active_a) {
                for (int c = lane_idx; c * 2 < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = a_px[c];
                }
            } else {
                for (int c = lane_idx; c * 2 < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = __float2half2_rn(0.0f);
                }
            }

            if (active_b) {
                for (int c = lane_idx; c * 2 < dim.out.c - dim.in.c; c += WARP_SIZE) {
                    out_px[c + dim.in.c / 2] = b_px[c];
                }
            } else {
                for (int c = lane_idx; c * 2 < dim.out.c - dim.in.c; c += WARP_SIZE) {
                    out_px[c + dim.in.c / 2] = __float2half2_rn(0.0f);
                }
            }
        } else {
            const scalar_t *a_px = &val_a[px_idx * dim.in.c];
            const scalar_t *b_px = &val_b[px_idx * (dim.out.c - dim.in.c)];
            scalar_t *out_px = &val_out[px_idx * dim.out.c];

            if (active_a) {
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = a_px[c];
                }
            } else {
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = __float2half(0.0f);
                }
            }

            if (active_b) {
                for (int c = lane_idx; c < dim.out.c - dim.in.c; c += WARP_SIZE) {
                    out_px[c + dim.in.c] = b_px[c];
                }
            } else {
                for (int c = lane_idx; c < dim.out.c - dim.in.c; c += WARP_SIZE) {
                    out_px[c + dim.in.c] = __float2half(0.0f);
                }
            }
        }
    }
}

template<typename scalar_t = float, int BLOCK_SIZE, int PIXELS_PER_BLOCK>
__global__ void deltacnn_sparse_mul_add_kernel(scalar_t *in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim) {
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);
    const int warp_idx = threadIdx.x / WARP_SIZE;

#ifdef ENABLE_METRICS
    const int lane_idx = threadIdx.x % WARP_SIZE;
    if (lane_idx == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t((end_idx - start_idx) * dim.in.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx - start_idx) * dim.in.c)); 
    }
#endif

    if (!checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask)) {
        return;
    }

    for (int px_idx = start_idx; px_idx < end_idx; ++px_idx) {
        bool active = mask[px_idx] != 0;

        if (!active) {
            continue;
        }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
        }
#endif

        const scalar_t *in_px = &in[px_idx * dim.in.c];
        scalar_t *out_px = &out[px_idx * dim.out.c];
        
        if (bias == nullptr) {
            for (int c = threadIdx.x; c < dim.in.c; c += BLOCK_SIZE) {
                out_px[c] = in_px[c] * scale[c];
            }
        } else {
            for (int c = threadIdx.x; c < dim.in.c; c += BLOCK_SIZE) {
                out_px[c] = in_px[c] * scale[c] + bias[c];
            }
        }
    }
}


template<typename scalar_t = float, int BLOCK_SIZE, int PIXELS_PER_BLOCK>
__global__ void deltacnn_sparse_mul_add_kernel_hp(scalar_t *in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim) {
    const int start_idx = blockIdx.x * PIXELS_PER_BLOCK;
    const int end_idx = min(start_idx + PIXELS_PER_BLOCK, dim.batch_size * dim.in.w * dim.in.h);
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

#ifdef ENABLE_METRICS
    if (lane_idx == 0) {
        atomicAdd(&d_metrics->n_vals_read_dense, uint64_t((end_idx - start_idx) * dim.out.c)); 
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t((end_idx - start_idx) * dim.out.c)); 
    }
#endif

    if (!checkIfAnyActive<PIXELS_PER_BLOCK, BLOCK_SIZE>(start_idx, end_idx, warp_idx, mask)) {
        return;
    }

    if (dim.in.c % 2 == 0) {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active = mask[px_idx] != 0;

            if (!active) {
                continue;
            }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
        }
#endif

            const half2 *in_px = reinterpret_cast<const half2*>(&in[px_idx * dim.in.c]);
            half2 *out_px = reinterpret_cast<half2*>(&out[px_idx * dim.out.c]);
            const half2 *scale2 = reinterpret_cast<const half2*>(scale);
            const half2 *bias2 = reinterpret_cast<const half2*>(bias);
            
            if (bias == nullptr) {
                for (int c = lane_idx; c * 2 < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = __hmul2(in_px[c], scale2[c]);
                }
            } else {
                for (int c = lane_idx; c * 2 < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = __hfma2(in_px[c], scale2[c], bias2[c]);
                }
            }
        }
    } else {
        for (int px_idx = start_idx + warp_idx; px_idx < end_idx; px_idx += n_warps) {
            bool active = mask[px_idx] != 0;

            if (!active) {
                continue;
            }
#ifdef ENABLE_METRICS
        if (lane_idx == 0) {
            atomicAdd(&d_metrics->n_vals_read, uint64_t(dim.in.c)); 
            atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.in.c)); 
        }
#endif

            const scalar_t *in_px = &in[px_idx * dim.in.c];
            scalar_t *out_px = &out[px_idx * dim.out.c];
            
            if (bias == nullptr) {
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = __hmul(in_px[c], scale[c]);
                }
            } else {
                for (int c = lane_idx; c < dim.in.c; c += WARP_SIZE) {
                    out_px[c] = __hfma(in_px[c], scale[c], bias[c]);
                }
            }
        }
    }
}

template<typename scalar_t, int threads, int pixels_per_block, int activation>
void activate_truncate_templates_vec_instructions(scalar_t* delta, scalar_t* prev_input, scalar_t* truncated, uint32_t* mask, float threshold, Dimensions dim, int truncation_mode, int blocks) {
    if (dim.in.c % 4 == 0 && dim.in.c > WARP_SIZE * 2) {
        deltacnn_activate_truncate_kernel<scalar_t, threads, pixels_per_block, activation, 4><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else if (dim.in.c % 2 == 0 && dim.in.c > WARP_SIZE) {
        deltacnn_activate_truncate_kernel<scalar_t, threads, pixels_per_block, activation, 2><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else {
        deltacnn_activate_truncate_kernel<scalar_t, threads, pixels_per_block, activation, 1><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    }
}

template<typename scalar_t, int threads, int pixels_per_block>
void activate_truncate_templates(scalar_t* delta, scalar_t* prev_input, scalar_t* truncated, uint32_t* mask, float threshold, Dimensions dim, int activation, int truncation_mode, int blocks) {
    if (activation <= 0) {
            activate_truncate_templates_vec_instructions<scalar_t, threads, pixels_per_block, 0>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode, blocks);
    } else if (activation == 1) {
            activate_truncate_templates_vec_instructions<scalar_t, threads, pixels_per_block, 1>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode, blocks);
    } else if (activation == 2) {
            activate_truncate_templates_vec_instructions<scalar_t, threads, pixels_per_block, 2>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode, blocks);
    } else if (activation == 3) {
            activate_truncate_templates_vec_instructions<scalar_t, threads, pixels_per_block, 3>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode, blocks);
    } else if (activation == 4) {
            activate_truncate_templates_vec_instructions<scalar_t, threads, pixels_per_block, 4>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode, blocks);
    } else if (activation == 5) {
            activate_truncate_templates_vec_instructions<scalar_t, threads, pixels_per_block, 5>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode, blocks);
    }
}


template<typename scalar_t = float>
void activate_truncate(scalar_t* delta, scalar_t* prev_input, scalar_t* truncated, uint32_t* mask, float threshold, Dimensions dim, int activation, int truncation_mode) {
    const int pixels = dim.batch_size * dim.in.w * dim.in.h;
    const int threads = 64;
    const int ppb_scale = 4;
    if (pixels > 40000 * ppb_scale) {
        const int pixels_per_block = 128;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 20000 * ppb_scale) {
        const int pixels_per_block = 64;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 10000 * ppb_scale) {
        const int pixels_per_block = 32;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 5000 * ppb_scale) {
        const int pixels_per_block = 16;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 1000 * ppb_scale) {
        const int pixels_per_block = 8;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else {
        const int pixels_per_block = 2;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    }
}

template<typename scalar_t, int threads, int pixels_per_block>
void activate_truncate_templates_hp(scalar_t*delta, scalar_t* prev_input, scalar_t* truncated, uint32_t *mask, float threshold, Dimensions dim, int activation, int truncation_mode, int blocks) {
    if (activation <= 0) {
        deltacnn_activate_truncate_hp_kernel<half, threads, pixels_per_block, 0><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else if (activation == 1) {
        deltacnn_activate_truncate_hp_kernel<half, threads, pixels_per_block, 1><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else if (activation == 2) {
        deltacnn_activate_truncate_hp_kernel<half, threads, pixels_per_block, 2><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else if (activation == 3) {
        deltacnn_activate_truncate_hp_kernel<half, threads, pixels_per_block, 3><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else if (activation == 4) {
        deltacnn_activate_truncate_hp_kernel<half, threads, pixels_per_block, 4><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    } else if (activation == 5) {
        deltacnn_activate_truncate_hp_kernel<half, threads, pixels_per_block, 5><<<blocks, threads>>>(delta, prev_input, truncated, mask, threshold, dim, truncation_mode);
    }
}

template<typename scalar_t = half>
void activate_truncate_hp(scalar_t*delta, scalar_t* prev_input, scalar_t* truncated, uint32_t *mask, float threshold, Dimensions dim, int activation, int truncation_mode) {
    const int pixels = dim.batch_size * dim.in.w * dim.in.h;
    const int threads = 64;
    const int ppb_scale = 4;
    if (pixels > 40000 * ppb_scale) {
        const int pixels_per_block = 128;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates_hp<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 20000 * ppb_scale) {
        const int pixels_per_block = 64;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates_hp<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 10000 * ppb_scale) {
        const int pixels_per_block = 32;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates_hp<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 5000 * ppb_scale) {
        const int pixels_per_block = 16;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates_hp<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else if (pixels > 1000 * ppb_scale) {
        const int pixels_per_block = 8;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates_hp<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    } else {
        const int pixels_per_block = 2;
        int blocks = divup(pixels, pixels_per_block);
        activate_truncate_templates_hp<scalar_t, threads, pixels_per_block>(delta, prev_input, truncated, mask, threshold, dim, activation, truncation_mode, blocks);
    }
}

template<typename scalar_t = float>
void prepare_diff_mask(scalar_t*input, scalar_t* prev_input, scalar_t* delta, uint32_t *mask, float threshold, Dimensions dim) {
    const int threads = 128;
    const int pixels_per_block = 32;
    int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
    deltacnn_prepare_diff_mask_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(input, prev_input, delta, mask, threshold, dim);
}

template<typename scalar_t = half>
void prepare_diff_mask_hp(scalar_t*input, scalar_t* prev_input, scalar_t* delta, uint32_t *mask, float threshold, Dimensions dim) {
    const int threads = 128;
    const int pixels_per_block = 32;
    int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
    deltacnn_prepare_diff_mask_hp_kernel<half, threads, pixels_per_block><<<blocks, threads>>>(input, prev_input, delta, mask, threshold, dim);
}

template<typename scalar_t, int threads, int pixels_per_block>
void add_tensors_templates(scalar_t* a, scalar_t* b, scalar_t* prev_out, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, scalar_t weight_a, scalar_t weight_b, Dimensions dim, int activation, bool dense_out, int blocks) {
    if (dense_out) {
        if (activation <= 0) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 0, true><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 1) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 1, true><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 2) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 2, true><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 3) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 3, true><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 4) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 4, true><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 5) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 5, true><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        }
    } else {
        if (activation <= 0) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 0, false><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 1) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 1, false><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 2) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 2, false><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 3) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 3, false><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 4) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 4, false><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        } else if (activation == 5) {
                deltacnn_sparse_add_tensors_kernel<scalar_t, threads, pixels_per_block, 5, false><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim);
        }
    }
}

template<typename scalar_t = float>
void sparse_add_tensors(scalar_t* a, scalar_t* b, scalar_t* prev_out, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, scalar_t weight_a, scalar_t weight_b, Dimensions dim, int activation, bool dense_out) {
     const int pixels = dim.batch_size * dim.in.w * dim.in.h;
    const int threads = 64;
    const int ppb_scale = 4;
    if (pixels > 40000 * ppb_scale) {
        const int pixels_per_block = 128;
        int blocks = divup(pixels, pixels_per_block);
        add_tensors_templates<scalar_t, threads, pixels_per_block>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out, blocks);
    } else if (pixels > 20000 * ppb_scale) {
        const int pixels_per_block = 64;
        int blocks = divup(pixels, pixels_per_block);
        add_tensors_templates<scalar_t, threads, pixels_per_block>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out, blocks);
    } else if (pixels > 10000 * ppb_scale) {
        const int pixels_per_block = 32;
        int blocks = divup(pixels, pixels_per_block);
        add_tensors_templates<scalar_t, threads, pixels_per_block>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out, blocks);
    } else if (pixels > 5000 * ppb_scale) {
        const int pixels_per_block = 16;
        int blocks = divup(pixels, pixels_per_block);
        add_tensors_templates<scalar_t, threads, pixels_per_block>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out, blocks);
    } else if (pixels > 1000 * ppb_scale) {
        const int pixels_per_block = 8;
        int blocks = divup(pixels, pixels_per_block);
        add_tensors_templates<scalar_t, threads, pixels_per_block>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out, blocks);
    } else {
        const int pixels_per_block = 2;
        int blocks = divup(pixels, pixels_per_block);
        add_tensors_templates<scalar_t, threads, pixels_per_block>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out, blocks);
    }
}

template<typename scalar_t = half>
void sparse_add_tensors_hp(scalar_t* a, scalar_t* b, scalar_t* prev_out, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, float weight_a, float weight_b, Dimensions dim, int activation, bool dense_out) {
    const int threads = 64;
    const int pixels_per_block = 8;
    int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
    deltacnn_sparse_add_tensors_hp_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(a , b, prev_out, out, mask_a, mask_b, mask_out, weight_a, weight_b, dim, activation, dense_out);
}

template<typename scalar_t>
void sparse_add_to_dense_tensor_sp(scalar_t* a, scalar_t* b, uint32_t *mask_a, Dimensions dim, int activation) {
    const int threads = 64;
    const int pixels = dim.batch_size * dim.in.w * dim.in.h;

    if (pixels > 10000) {
        const int pixels_per_block = 32;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    } else if (pixels > 5000) {
        const int pixels_per_block = 16;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    } else if (pixels > 1000) {
        const int pixels_per_block = 8;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    } else {
        const int pixels_per_block = 2;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_sp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    }
}

template<typename scalar_t>
void sparse_add_to_dense_tensor_hp(scalar_t* a, scalar_t* b, uint32_t *mask_a, Dimensions dim, int activation) {
    const int threads = 64;
    const int pixels = dim.batch_size * dim.in.w * dim.in.h;
    const int ppb_scale = 4;

    if (pixels > 30000 * ppb_scale) {
        const int pixels_per_block = 64;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    }
    else if (pixels > 10000 * ppb_scale) {
        const int pixels_per_block = 32;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    } else if (pixels > 5000 * ppb_scale) {
        const int pixels_per_block = 16;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    } else if (pixels > 1000 * ppb_scale) {
        const int pixels_per_block = 8;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    } else {
        const int pixels_per_block = 2;
        const int blocks = divup(pixels, pixels_per_block);
        if (activation <= 0) {
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, false><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        } else {        
            deltacnn_sparse_add_to_dense_tensor_kernel_hp<scalar_t, threads, pixels_per_block, true><<<blocks, threads>>>(a, b, mask_a, dim, activation);
        }
    }
}

template<typename scalar_t, int pixels_per_block>
void sparse_mul_add_block_size_templates(scalar_t* in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim) { 
    if (dim.out.c <= 32 || pixels_per_block <= 1) {
        const int threads = 32;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (dim.out.c <= 64 || pixels_per_block <= 2) {
        const int threads = 64;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (dim.out.c <= 128 || pixels_per_block <= 4) {
        const int threads = 128;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (dim.out.c <= 256 || pixels_per_block <= 8) {
        const int threads = 256;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (dim.out.c <= 512 || pixels_per_block <= 16) {
        const int threads = 512;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else {
        const int threads = 1024;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    }
}

template<typename scalar_t>
void sparse_mul_add(scalar_t* in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim) {
    int pixels = dim.batch_size * dim.in.w * dim.in.h;
    if (pixels > 100000) {
        sparse_mul_add_block_size_templates<scalar_t, 256>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 50000) {
        sparse_mul_add_block_size_templates<scalar_t, 128>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 25000) {
        sparse_mul_add_block_size_templates<scalar_t, 64>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 10000) {
        sparse_mul_add_block_size_templates<scalar_t, 32>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 5000) {
        sparse_mul_add_block_size_templates<scalar_t, 16>(in, mask, out, mask_out, scale, bias, dim);
    }  else if (pixels > 2500) {
        sparse_mul_add_block_size_templates<scalar_t, 8>(in, mask, out, mask_out, scale, bias, dim);
    } else {
        sparse_mul_add_block_size_templates<scalar_t, 2>(in, mask, out, mask_out, scale, bias, dim);
    }
}
template<typename scalar_t>
void sparse_mul_add_hp(scalar_t* in, uint32_t *mask, scalar_t *out, uint32_t *mask_out, scalar_t *scale, scalar_t *bias, Dimensions dim) {
    int pixels = dim.batch_size * dim.in.w * dim.in.h;

    const int ppb_scale = 4;
    const int threads = 64;
    if (pixels > 100000 * ppb_scale) {
        const int pixels_per_block = 256;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 50000 * ppb_scale) {
        const int pixels_per_block = 128;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 25000 * ppb_scale) {
        const int pixels_per_block = 64;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 10000 * ppb_scale) {
        const int pixels_per_block = 32;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else if (pixels > 5000 * ppb_scale) {
        const int pixels_per_block = 16;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    }  else if (pixels > 2500 * ppb_scale) {
        const int pixels_per_block = 8;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    } else {
        const int pixels_per_block = 2;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_mul_add_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(in, mask, out, mask_out, scale, bias, dim);
    }
}

template<typename scalar_t = float>
void sparse_upsample(scalar_t* in, scalar_t* out, uint32_t *mask_in, uint32_t *mask_out, Dimensions dim, int scale) {
    const int threads = 64;
    const int pixels_per_block = 2;
    if (scale == 2) {
        const int scale = 2;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else if (scale == 4) {
        const int scale = 4;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else if (scale == 8) {
        const int scale = 8;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else if (scale == 16) {
        const int scale = 16;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else {
        printf("Upscale factors other than 2, 4, 8 and 16 not supported, got %i\n", scale);
        throw "Upscale factors other than 2, 4, 8 and 16 not supported";
    }
}

template<typename scalar_t = half>
void sparse_upsample_hp(scalar_t* in, scalar_t* out, uint32_t *mask_in, uint32_t *mask_out, Dimensions dim, int scale) {
    const int threads = 128;
    const int pixels_per_block = 32;
    if (scale == 2) {
        const int scale = 2;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_hp_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else if (scale == 4) {
        const int scale = 4;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_hp_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else if (scale == 8) {
        const int scale = 8;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_hp_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else if (scale == 16) {
        const int scale = 16;
        int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);
        deltacnn_sparse_upsample_hp_kernel<scalar_t, threads, pixels_per_block, scale><<<blocks, threads>>>(in, out, mask_in, mask_out, dim);
    } else {
        printf("Upscale factors other than 2, 4, 8 and 16 not supported, got %i\n", scale);
        throw "Upscale factors other than 2, 4, 8 and 16 not supported";
    }
}

template<typename scalar_t>
void sparse_concatenate(scalar_t* a, scalar_t* b, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, Dimensions dim) {
    const int threads = 64;
    const int pixels_per_block = 2;
    int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);

    deltacnn_sparse_concatenate_kernel<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(a, b, out, mask_a, mask_b, mask_out, dim);
}

template<typename scalar_t>
void sparse_concatenate_hp(scalar_t* a, scalar_t* b, scalar_t* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, Dimensions dim) {
    const int threads = 64;
    const int pixels_per_block = 2;
    int blocks = divup(dim.batch_size * dim.in.w * dim.in.h, pixels_per_block);

    deltacnn_sparse_concatenate_kernel_hp<scalar_t, threads, pixels_per_block><<<blocks, threads>>>(a, b, out, mask_a, mask_b, mask_out, dim);
}

template<typename scalar_t, int pool_mode, int threads>
void sparse_pool_templates(scalar_t* input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    const int pixelsPerBlock = 1;
    const int out_channels_per_block = threads;
    uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlock*config.dilation[0]) * divup(dim.out.w, pixelsPerBlock*config.dilation[1]) * config.dilation[0] * config.dilation[1];

    if (config.kernel_size[0] == 2 && config.kernel_size[1] == 2) {
        const int kernel_size = 2;
        if (config.dilation[0] == 1 && config.dilation[1] == 1) {
            const int dilation = 1;
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else {
                printf("Stride other than 1x1 and 2x2 not supported for 2x2 pooling, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1 and 2x2 not supported for 2x2 pooling";
            }
        } else {
            deltacnn_sparse_pooling_sp_single_element<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    }
    else if (config.kernel_size[0] == 3 && config.kernel_size[1] == 3) {
        const int kernel_size = 3;

        if (config.dilation[0] == 1 && config.dilation[1] == 1) {
            const int dilation = 1;
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else if (config.stride[0] == 3 && config.stride[1] == 3) {
                const int stride = 3;
                deltacnn_sparse_pooling_sp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else {
                printf("Stride other than 1x1, 2x2 and 3x3 not supported for 3x3 pooling, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1, 2x2 and 3x3 not supported for 3x3 pooling";
            }
        } else {
            deltacnn_sparse_pooling_sp_single_element<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    }
    else if (config.kernel_size[0] == 5 && config.kernel_size[1] == 5) {
        const int kernel_size = 5;
        if (config.dilation[0] == 1 && config.dilation[1] == 1) {
            const int dilation = 1;
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 3 && config.stride[1] == 3) {
                const int stride = 3;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 4 && config.stride[1] == 4) {
                const int stride = 4;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 5 && config.stride[1] == 5) {
                const int stride = 5;
                deltacnn_sparse_pooling_sp_less_registers<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else {
                printf("Stride other than 1x1, 2x2, 3x3, 4x4 and 5x5 not supported for 5x5 pooling, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1, 2x2, 3x3, 4x4 and 5x5 not supported for 5x5 pooling";
            }
        } else {
            deltacnn_sparse_pooling_sp_single_element<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    } else {
        deltacnn_sparse_pooling_sp_single_element<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
            input, prev_input, out, mask, out_mask, dim, config);
    }
}

template<typename scalar_t, int pool_mode>
void sparse_pool_thread_templates(scalar_t* input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    // TODO refactor this mess. reduce set of implementations to reduce risk of bugs
    if (dim.out.w == 1 && dim.out.h == 1 && config.kernel_size[0] > 16 && config.kernel_size[1] > 16) {
        if (dim.out.c <= 32) {
            const int threads = 32;
            const int pixelsPerBlock = 32;
            const int out_channels_per_thread = 1;
            const int out_channels_per_block = threads * out_channels_per_thread;
            const int blocks = divup(dim.in.h * dim.in.w, pixelsPerBlock);
            dim3 gridDim(blocks, dim.batch_size, divup(dim.out.c, out_channels_per_block));

            deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px<scalar_t, threads, pixelsPerBlock, out_channels_per_thread, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        } else if (dim.out.c <= 64) {
            const int threads = 64;
            const int pixelsPerBlock = 16;
            const int out_channels_per_thread = 1;
            const int out_channels_per_block = threads * out_channels_per_thread;
            const int blocks = divup(dim.in.h * dim.in.w, pixelsPerBlock);
            dim3 gridDim(blocks, dim.batch_size, divup(dim.out.c, out_channels_per_block));
            
            deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px<scalar_t, threads, pixelsPerBlock, out_channels_per_thread, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        } else {
            const int threads = 64;
            const int pixelsPerBlock = 8;
            const int out_channels_per_thread = 2;
            const int out_channels_per_block = threads * out_channels_per_thread;
            const int blocks = divup(dim.in.h * dim.in.w, pixelsPerBlock);
            dim3 gridDim(blocks, dim.batch_size, divup(dim.out.c, out_channels_per_block));

            deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px<scalar_t, threads, pixelsPerBlock, out_channels_per_thread, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    }
    else if (config.kernel_size[0]*config.kernel_size[1] >= WARP_SIZE) {
        // special mode for very large kernels
        const int threads = WARP_SIZE;
        uint32_t blocks = dim.batch_size * divup(dim.out.h, config.dilation[0]) * divup(dim.out.w, config.dilation[1]) * config.dilation[0] * config.dilation[1];
        if (dim.out.c <= 32) {
            const int out_channels_per_block = 1;
            dim3 gridDim(blocks, 1, divup(dim.out.c, out_channels_per_block));
            deltacnn_sparse_pooling_sp_warp_per_channel<scalar_t, threads, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        } else if (dim.out.c <= 64) {
            const int out_channels_per_block = 2;
            dim3 gridDim(blocks, 1, divup(dim.out.c, out_channels_per_block));
            deltacnn_sparse_pooling_sp_warp_per_channel<scalar_t, threads, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        } else {
            const int out_channels_per_block = 4;
            dim3 gridDim(blocks, 1, divup(dim.out.c, out_channels_per_block));
            deltacnn_sparse_pooling_sp_warp_per_channel<scalar_t, threads, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    }
    else if (dim.out.c <= 32) {
        sparse_pool_templates<scalar_t, pool_mode, 32>(input, prev_input, out, mask, out_mask, dim, config);
    } else if (dim.out.c <= 64) {
        sparse_pool_templates<scalar_t, pool_mode, 64>(input, prev_input, out, mask, out_mask, dim, config);
    } else if (dim.out.c <= 128) {
        sparse_pool_templates<scalar_t, pool_mode, 128>(input, prev_input, out, mask, out_mask, dim, config);
    } else if (dim.out.c <= 256) {
        sparse_pool_templates<scalar_t, pool_mode, 256>(input, prev_input, out, mask, out_mask, dim, config);
    } else if (dim.out.c <= 384) {
        sparse_pool_templates<scalar_t, pool_mode, 384>(input, prev_input, out, mask, out_mask, dim, config);
    } else {
        sparse_pool_templates<scalar_t, pool_mode, 512>(input, prev_input, out, mask, out_mask, dim, config);
    }
}

template<typename scalar_t>
void sparse_pool(scalar_t* input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config, int pooling_mode) {
    if (pooling_mode == 0) {
        sparse_pool_thread_templates<scalar_t, 0>(input, prev_input, out, mask, out_mask, dim, config);
    } else if (pooling_mode == 1) {
        sparse_pool_thread_templates<scalar_t, 1>(input, prev_input, out, mask, out_mask, dim, config);
    } else {
        printf("Pooling modes other than 0 (max) and 1 (avg) not implemented, got %i\n", pooling_mode);
        throw "Pooling modes other than 0 (max) and 1 (avg) not implemented";
    }
}

template<typename scalar_t, int pool_mode>
void sparse_pool_thread_templates_hp(scalar_t *input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    const int pixelsPerBlock = 1;
    const int threads = 64;
    const int out_channels_per_block = threads;
    uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlock*config.dilation[0]) * divup(dim.out.w, pixelsPerBlock*config.dilation[1]) * config.dilation[0] * config.dilation[1];

    if (dim.out.w == 1 && dim.out.h == 1) {
        float *fout = (float*) out;
        if (dim.out.c <= 32) {
            const int threads = 32;
            const int pixelsPerBlock = 32;
            const int out_channels_per_thread = 1;
            const int out_channels_per_block = threads * out_channels_per_thread;
            const int blocks = divup(dim.in.h * dim.in.w, pixelsPerBlock);
            dim3 gridDim(blocks, dim.batch_size, divup(dim.out.c, out_channels_per_block));

            deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px_hp<scalar_t, threads, pixelsPerBlock, out_channels_per_thread, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, fout, mask, out_mask, dim, config);
        } else if (dim.out.c <= 64) {
            const int threads = 64;
            const int pixelsPerBlock = 16;
            const int out_channels_per_thread = 1;
            const int out_channels_per_block = threads * out_channels_per_thread;
            const int blocks = divup(dim.in.h * dim.in.w, pixelsPerBlock);
            dim3 gridDim(blocks, dim.batch_size, divup(dim.out.c, out_channels_per_block));
            
            deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px_hp<scalar_t, threads, pixelsPerBlock, out_channels_per_thread, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, fout, mask, out_mask, dim, config);
        } else {
            const int threads = 64;
            const int pixelsPerBlock = 8;
            const int out_channels_per_thread = 2;
            const int out_channels_per_block = threads * out_channels_per_thread;
            const int blocks = divup(dim.in.h * dim.in.w, pixelsPerBlock);
            dim3 gridDim(blocks, dim.batch_size, divup(dim.out.c, out_channels_per_block));

            deltacnn_sparse_pooling_sp_block_per_in_pixel_single_out_px_hp<scalar_t, threads, pixelsPerBlock, out_channels_per_thread, out_channels_per_block, true, false, pool_mode><<<gridDim, threads>>>(
                input, prev_input, fout, mask, out_mask, dim, config);
        }
    }
    else if (config.kernel_size[0] == 2 && config.kernel_size[1] == 2) {
        const int kernel_size = 2;
        if (config.dilation[0] == 1 && config.dilation[1] == 1) {
            const int dilation = 1;
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else {
                printf("Stride other than 1x1 and 2x2 not supported for 2x2 pooling, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1 and 2x2 not supported for 2x2 pooling";
            }
        } else {
            deltacnn_sparse_pooling_sp_single_element_hp<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    }
    else if (config.kernel_size[0] == 3 && config.kernel_size[1] == 3) {
        const int kernel_size = 3;

        if (config.dilation[0] == 1 && config.dilation[1] == 1) {
            const int dilation = 1;
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else if (config.stride[0] == 3 && config.stride[1] == 3) {
                const int stride = 3;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            }
            else {
                printf("Stride other than 1x1, 2x2 and 3x3 not supported for 3x3 pooling, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1, 2x2 and 3x3 not supported for 3x3 pooling";
            }
        } else {
            deltacnn_sparse_pooling_sp_single_element_hp<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    }
    else if (config.kernel_size[0] == 5 && config.kernel_size[1] == 5) {
        const int kernel_size = 5;
        if (config.dilation[0] == 1 && config.dilation[1] == 1) {
            const int dilation = 1;
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 3 && config.stride[1] == 3) {
                const int stride = 3;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 4 && config.stride[1] == 4) {
                const int stride = 4;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else if (config.stride[0] == 5 && config.stride[1] == 5) {
                const int stride = 5;
                deltacnn_sparse_pooling_sp_less_registers_hp<scalar_t, kernel_size, pixelsPerBlock, threads, out_channels_per_block, true, true, stride, dilation, pool_mode><<<blocks, threads>>>(
                    input, prev_input, out, mask, out_mask, dim, config);
            } else {
                printf("Stride other than 1x1, 2x2, 3x3, 4x4 and 5x5 not supported for 5x5 pooling, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1, 2x2, 3x3, 4x4 and 5x5 not supported for 5x5 pooling";
            }
        } else {
            deltacnn_sparse_pooling_sp_single_element_hp<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
                input, prev_input, out, mask, out_mask, dim, config);
        }
    } else {
        deltacnn_sparse_pooling_sp_single_element_hp<scalar_t, threads, out_channels_per_block, true, true, pool_mode><<<blocks, threads>>>(
            input, prev_input, out, mask, out_mask, dim, config);
    }
}

template<typename scalar_t>
void sparse_pool_hp(scalar_t* input, scalar_t* prev_input, scalar_t* out, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config, int pooling_mode) {
    if (pooling_mode == 0) {
        sparse_pool_thread_templates_hp<scalar_t, 0>(input, prev_input, out, mask, out_mask, dim, config);
    } else if (pooling_mode == 1) {
        sparse_pool_thread_templates_hp<scalar_t, 1>(input, prev_input, out, mask, out_mask, dim, config);
    } else {
        printf("Pooling modes other than 0 (max) and 1 (avg) not implemented, got %i\n", pooling_mode);
        throw "Pooling modes other than 0 (max) and 1 (avg) not implemented";
    }
}

template void activate_truncate<float>(float *delta, float *prev_input, float *truncated, uint32_t *mask, float threshold, Dimensions dim, int activation, int truncation_mode);
template void activate_truncate_hp<half>(half *delta, half *prev_input, half *truncated, uint32_t *mask, float threshold, Dimensions dim, int activation, int truncation_mode);
template void prepare_diff_mask<float>(float *input, float *prev_input, float *delta, uint32_t *mask, float threshold, Dimensions dim);
template void prepare_diff_mask_hp<half>(half *input, half *prev_input, half *delta, uint32_t *mask, float threshold, Dimensions dim);
template void sparse_add_tensors<float>(float* a, float* b, float* prev_out, float* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, float weight_a, float weight_b, Dimensions dim, int activation, bool dense_out);
template void sparse_add_tensors_hp<half>(half* a, half* b, half* prev_out, half* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, float weight_a, float weight_b, Dimensions dim, int activation, bool dense_out);
template void sparse_add_to_dense_tensor_sp<float>(float* a, float* b, uint32_t *mask_a, Dimensions dim, int activation);
template void sparse_add_to_dense_tensor_hp<half>(half* a, half* b, uint32_t *mask_a, Dimensions dim, int activation);
template void sparse_upsample<float>(float* in, float* out, uint32_t *mask_in, uint32_t *mask_out, Dimensions dim, int scale);
template void sparse_upsample_hp<half>(half* in, half* out, uint32_t *mask_in, uint32_t *mask_out, Dimensions dim, int scale);
template void sparse_concatenate<float>(float* a, float* b, float* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, Dimensions dim);
template void sparse_concatenate_hp<half>(half* a, half* b, half* out, uint32_t *mask_a, uint32_t *mask_b, uint32_t *mask_out, Dimensions dim);
template void sparse_pool<float>(float* input, float* prev_input, float* out, uint32_t *mask, uint32_t* out_mask, Dimensions dim, ConvConfig config, int pooling_mode);
template void sparse_pool_hp<half>(half* input, half* prev_input, half* out, uint32_t *mask, uint32_t* out_mask, Dimensions dim, ConvConfig config, int pooling_mode);
template void sparse_mul_add<float>(float* in, uint32_t *mask, float *out, uint32_t *mask_out, float *scale, float *bias, Dimensions dim);
template void sparse_mul_add_hp<half>(half* in, uint32_t *mask, half *out, uint32_t *mask_out, half *scale, half *bias, Dimensions dim);