// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "conv_kernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Utility.cuh"

#define divup(a, b) (((a) + (b) - 1) / (b))

__device__ DCMetrics* d_metrics;

void init_d_metrics_conv_kernels() {
#ifdef ENABLE_METRICS
    copy_performance_metrics_to_gpu(d_metrics);
#endif
}


template<int pixelsPerBlockX, int pixelsPerBlockY, int OUT_CHANNELS_PER_BLOCK, int STRIDE, bool FULL_DEPTH, bool ENABLE_DILATION>
__device__ __forceinline__ void calc_tile_indices(int& tile_start_out_y, int& tile_start_out_x, int& tile_start_in_y, int& tile_start_in_x, int& tile_start_z, int& batch, const ConvConfig& config, const Dimensions& dim) {
    if (ENABLE_DILATION) {
        tile_start_out_y = (blockIdx.y / config.dilation[0]) * pixelsPerBlockY * config.dilation[0] + (blockIdx.y % config.dilation[0]);
        tile_start_out_x = (blockIdx.x / config.dilation[1]) * pixelsPerBlockX * config.dilation[1] + (blockIdx.x % config.dilation[1]);
        tile_start_in_y = tile_start_out_y * STRIDE - config.padding[0];
        tile_start_in_x = tile_start_out_x * STRIDE - config.padding[1];

    } else {
        tile_start_out_y = blockIdx.y * pixelsPerBlockY;
        tile_start_out_x = blockIdx.x * pixelsPerBlockX;
        tile_start_in_y = tile_start_out_y * STRIDE - config.padding[0];
        tile_start_in_x = tile_start_out_x * STRIDE - config.padding[1];
    }
    if (FULL_DEPTH) {
        tile_start_z = 0;
        batch = blockIdx.z;
    } else {
        const int blocksPerBatch = divup(dim.out.c, OUT_CHANNELS_PER_BLOCK);
        tile_start_z = (blockIdx.z % blocksPerBatch) * OUT_CHANNELS_PER_BLOCK;
        batch = blockIdx.z / blocksPerBatch;
    }
}

template<int BLOCK_SIZE, int n_in_px, int w_in, bool ENABLE_DILATION, int KERNEL_SIZE=3, int STRIDE=1>
__device__ __forceinline__ void load_mask(const int tile_start_in_y, const int tile_start_in_x, const int lane_idx, const uint32_t* batch_mask, uint32_t* s_mask, uint64_t& t_mask, uint32_t& density, const Dimensions& dim, const ConvConfig& config) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    
    // because in a 1x1 conv, we can directly skip input elements which we can't do for larger convs
    const int STRIDE_FACTOR = KERNEL_SIZE == 1 ? STRIDE : 1;

    for (int i = threadIdx.x; i < n_in_px; i += BLOCK_SIZE) {
        int y = tile_start_in_y + (i / w_in) * STRIDE_FACTOR * DILATION_Y;
        int x = tile_start_in_x + (i % w_in) * STRIDE_FACTOR * DILATION_X;
        if (y >= 0 && y < dim.in.h && x >= 0 && x < dim.in.w) {
            const int mask_idx = y * dim.in.w + x;
            s_mask[i] = batch_mask != nullptr ? batch_mask[mask_idx] : 1;
        } else {
            s_mask[i] = 0;
        }
    }
    __syncthreads();

    if (n_in_px <= 64) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int px_idx = lane_idx + i * WARP_SIZE;
            uint32_t mask = px_idx < n_in_px ? s_mask[px_idx] : 0;
            mask = __ballot_sync(0xFFFFFFFF, mask);
            if (i == 0) {
                t_mask = uint64_t(mask);
            } else {
                t_mask += (uint64_t(mask) << 32); 
            }
        }
        density = __popcll(t_mask);
    } else {
        density = 0;
        #pragma unroll
        for (int i = 0; i < divup(n_in_px, WARP_SIZE); ++i) {
            int px_idx = lane_idx + i * WARP_SIZE;
            uint32_t mask = px_idx < n_in_px ? s_mask[px_idx] : 0;
            mask = __ballot_sync(0xFFFFFFFF, mask);
            density += __popc(mask);
        }
    }

#ifdef ENABLE_METRICS
        if (threadIdx.x == 0 && blockIdx.z == 0) {
            atomicAdd(&d_metrics->n_tiles, uint64_t(1)); 
            atomicAdd(&d_metrics->n_inputs, uint64_t(n_in_px)); 
            atomicAdd(&d_metrics->active_input_histogram[density], uint64_t(1)); 

            if (density > 0) {
                atomicAdd(&d_metrics->n_active_tiles, uint64_t(1)); 
                atomicAdd(&d_metrics->n_active_inputs, uint64_t(density)); 
                
                if (DCMetrics::track_filter_reads) {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(density * dim.in.c + KERNEL_SIZE*KERNEL_SIZE*dim.in.c*dim.out.c/config.groups)); 
                } else {
                    atomicAdd(&d_metrics->n_vals_read, uint64_t(density * dim.in.c)); 
                }
            }
        }
#endif
}

template<int BLOCK_SIZE, int n_in_px, int w_in, bool ENABLE_DILATION>
__device__ __noinline__ void load_mask_noinline(const int tile_start_in_y, const int tile_start_in_x, const int lane_idx, const uint32_t* batch_mask, uint32_t* s_mask, uint64_t& t_mask, uint32_t& density, const Dimensions& dim, const ConvConfig& config) {
    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, s_mask, t_mask, density, dim, config);
}


template<int STRIDE, int w_in, int KERNEL_SIZE=3>
__device__ __forceinline__ bool is_out_px_active(const int x, const int y, const uint64_t t_mask) {
    uint64_t mask;
    if (KERNEL_SIZE == 3) {
        if (w_in == 3){
            mask = 0b1111111111LLU;
        } else if (w_in == 4) {
            mask = 0b11101110111LLU;
        } else if (w_in == 5) {
            mask = 0b1110011100111LLU;
        } else if (w_in == 6) {
            mask = 0b111000111000111LLU;
        } else if (w_in == 7) {
            mask = 0b11100001110000111LLU;
        } else if (w_in == 8) {
            mask = 0b1110000011100000111LLU;
        } else if (w_in == 9) {
            mask = 0b111000000111000000111LLU;
        } else if (w_in == 10) {
            mask = 0b11100000001110000000111LLU;
        } else {
            mask = 0;
        }
    } else if (KERNEL_SIZE == 1) {
        mask = 1;
    } else {
        mask = 0;
    }

    int offset = y * STRIDE * w_in + x * STRIDE;
    mask = mask << offset;
    bool valid = (t_mask & mask) != 0;
    return valid;
}

template<int w_in, int n_in_px>
__device__ __forceinline__ bool is_px_mask_set(const int x, const int y, const uint64_t t_mask, const uint32_t *s_mask) {
    return n_in_px <= 64 ? ((t_mask & (1LLU << (y * w_in + x))) != 0) : (s_mask[y * w_in + x] != 0);
}

template<int BLOCK_SIZE, int pixelsPerBlockX, int n_pixels_out, int w_in, int n_in_px, int STRIDE, bool ENABLE_DILATION, int KERNEL_SIZE=3> 
__device__ __forceinline__ void write_mask(uint32_t* out_mask, const int batch, const uint64_t t_mask, const uint32_t *s_mask, const int density, const int tile_start_out_y, const int tile_start_out_x, const Dimensions &dim, const ConvConfig &config) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];

    for (int out_px = threadIdx.x; out_px < n_pixels_out; out_px += BLOCK_SIZE) {
        int p_y = out_px / pixelsPerBlockX;
        int p_x = out_px % pixelsPerBlockX;
        int out_y = p_y * DILATION_Y + tile_start_out_y;
        int out_x = p_x * DILATION_X + tile_start_out_x; 
        if (out_y >= dim.out.h || out_x >= dim.out.w)
            continue;

        if ((n_in_px <= 64 && t_mask != 0LLU) || (n_in_px > 64 && density != 0)) {
            bool updated = false;
#ifdef ENABLE_METRICS
            int active_inputs = 0;
#endif
            if (KERNEL_SIZE == 1) {
                updated = is_px_mask_set<w_in, n_in_px>(p_x, p_y, t_mask, s_mask);
#ifdef ENABLE_METRICS
                if (updated) {
                    ++active_inputs;
                }
#endif
            } else {
                for (int y = 0; y < KERNEL_SIZE; y++) {
                    for (int x = 0; x < KERNEL_SIZE; x++) {
                        const bool valid = is_px_mask_set<w_in, n_in_px>(p_x*STRIDE + x, p_y*STRIDE + y, t_mask, s_mask);
                        updated |= valid;
#ifdef ENABLE_METRICS
                        if (valid) {
                            ++active_inputs;
                        }
#endif
                    }
                }
            }
            batch_out_mask[out_y * dim.out.w + out_x] = updated ? 1:0;
#ifdef ENABLE_METRICS
            if (density > 0) {
                if (config.groups == 1 && !(KERNEL_SIZE == 3 && density <= 4 && config.stride[0] == 1 && config.dilation[0] == 1)) {
                    atomicAdd(&d_metrics->n_active_flops, uint64_t(KERNEL_SIZE * KERNEL_SIZE * dim.in.c * dim.out.c / config.groups)); 
                } else {
                    // depthwise and 3x3 convolutions with <= 4 active inputs use sub-tile sparsity 
                    atomicAdd(&d_metrics->n_active_flops, uint64_t(active_inputs * dim.in.c * dim.out.c / config.groups)); 
                }
            }
            if (updated) {
                atomicAdd(&d_metrics->n_theoretical_flops, uint64_t(active_inputs * dim.in.c * dim.out.c / config.groups)); 
                atomicAdd(&d_metrics->n_vals_written, uint64_t(dim.out.c)); 
            }
#endif
        } else {
            batch_out_mask[out_y * dim.out.w + out_x] = 0;
        }
    }
#ifdef ENABLE_METRICS
    if (threadIdx.x == 0 && blockIdx.z == 0) {
        atomicAdd(&d_metrics->n_dense_flops, uint64_t(n_pixels_out * (KERNEL_SIZE * KERNEL_SIZE * dim.in.c * dim.out.c / config.groups))); 
    }
#endif
}

template<typename scalar_t, int BLOCK_SIZE, int n_pixels_out, int pixelsPerBlockX, int OUT_CHANNELS_PER_BLOCK, bool FULL_DEPTH, bool ENABLE_DILATION>
__device__ __forceinline__ void set_out_zero(scalar_t* batch_out, const scalar_t* bias, const int tile_start_z, const int tile_start_out_y, const int tile_start_out_x, const Dimensions& dim, const ConvConfig& config) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    const int c_per_block = FULL_DEPTH ? dim.out.c : min(OUT_CHANNELS_PER_BLOCK, dim.out.c - tile_start_z);

    for (int out_idx = threadIdx.x; out_idx < n_pixels_out * c_per_block; out_idx += BLOCK_SIZE) {
        int out_px = out_idx / c_per_block;
        int out_c = (out_idx % c_per_block) + tile_start_z;
        int p_y = out_px / pixelsPerBlockX;
        int p_x = out_px % pixelsPerBlockX;
        int out_y = p_y * DILATION_Y + tile_start_out_y;
        int out_x = p_x * DILATION_X + tile_start_out_x; 
        if (out_y >= dim.out.h || out_x >= dim.out.w)
            continue;

        scalar_t out_val = bias == nullptr ? 0.0f : bias[out_c];
        batch_out[(out_y * dim.out.w + out_x) * dim.out.c + out_c] = out_val;
    }
}

template<typename scalar_t, int BLOCK_SIZE, int n_pixels_out, int pixelsPerBlockX, int OUT_CHANNELS_PER_BLOCK, bool FULL_DEPTH, bool ENABLE_DILATION>
__device__ __forceinline__ void set_out_zero_hp(scalar_t* batch_out, const scalar_t* bias, const int tile_start_z, const int tile_start_out_y, const int tile_start_out_x, const Dimensions& dim, const ConvConfig& config) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    const int c_per_block = FULL_DEPTH ? dim.out.c : min(OUT_CHANNELS_PER_BLOCK, dim.out.c - tile_start_z);

    for (int out_idx = threadIdx.x; out_idx < n_pixels_out * c_per_block; out_idx += BLOCK_SIZE) {
        int out_px = out_idx / c_per_block;
        int out_c = (out_idx % c_per_block) + tile_start_z;
        int p_y = out_px / pixelsPerBlockX;
        int p_x = out_px % pixelsPerBlockX;
        int out_y = p_y * DILATION_Y + tile_start_out_y;
        int out_x = p_x * DILATION_X + tile_start_out_x; 
        if (out_y >= dim.out.h || out_x >= dim.out.w)
            continue;

        scalar_t out_val = bias == nullptr ? __float2half(0.0f) : bias[out_c];
        batch_out[(out_y * dim.out.w + out_x) * dim.out.c + out_c] = out_val;
    }
}

template<typename scalar_t = float, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void deltacnn_3x3_sp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    const Dimensions dim,
    const ConvConfig config
) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    const int in_row_vals = dim.in.w * dim.in.c;
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int KERNEL_SIZE = 3;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION>(
        tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim
    );

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);

    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2;
    const int n_in_px = w_in * h_in;
    const int n_in_px_aligned = divup(n_in_px, 4) * 4;

    const int sparse_mode_max_elements = 4;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int sub_warp_idx = lane_idx / 8;
    const int sub_warp_lane_idx = lane_idx % 8;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    struct SparseSMEM {
        scalar_t s_in[sparse_mode_max_elements][WARP_SIZE];
        scalar_t s_out[n_pixels_out][BLOCK_SIZE];
    };
    union SMEM {
        SparseSMEM sparse;
        scalar_t dense_s_in[n_in_px_aligned][WARP_SIZE];
    };

    __shared__ SMEM smem;
    __shared__ uint32_t s_mask[n_in_px];
    uint64_t t_mask = 0LLU;
    uint32_t density = 0;

#ifdef ENABLE_METRICS
    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("&d_metrics=%p\n", d_metrics);
    //     printf("&d_metrics.vals_read_dense = %p\n", &d_metrics->n_vals_read_dense);
    //     printf("&d_metrics.vals_written_dense = %p\n", &d_metrics->n_vals_written_dense);
    //     ++d_metrics->n_active_flops;
    //     printf("d_metrics.n_active_flops = %i\n", d_metrics->n_active_flops);
    // } 
    // return;

    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);


    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if ((n_in_px <= 64 && t_mask == 0LLU) || (n_in_px < 64 && density == 0)) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, ENABLE_DILATION>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    // TODO fix dilation and striding for sparse version
    if (SUB_TILE_SPARSITY && density <= sparse_mode_max_elements && STRIDE == 1 && DILATION_X == 1 && DILATION_Y == 1) {
#ifdef ENABLE_METRICS
        if (threadIdx.x == 0 && blockIdx.z == 0) {
            atomicAdd(&d_metrics->n_tiles_sparse_mode, 1);
        }
#endif
        int set_elements[sparse_mode_max_elements];
        int last_element_idx = -1;
        #pragma unroll
        for (int update_idx = 0; update_idx < sparse_mode_max_elements; ++update_idx) {
            set_elements[update_idx] = -1;
            for (++last_element_idx; last_element_idx < n_in_px; ++last_element_idx) {
                if (s_mask[last_element_idx] != 0) {
                    set_elements[update_idx] = last_element_idx;
                    break;
                }
            }
        }

        for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE) {
            const int out_c = out_c_off + threadIdx.x;
            const scalar_t t_bias = bias == nullptr || out_c >= dim.out.c ? 0.0f : bias[out_c];
            for (int i = 0; i < n_pixels_out; ++i) {
                smem.sparse.s_out[i][threadIdx.x] = t_bias;
            }

            for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE) {
                __syncthreads();

                #pragma unroll
                for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                    if (set_elements[element_idx] == -1 || element_idx % n_warps != warp_idx) {
                        continue;
                    }

                    const int px_idx = set_elements[element_idx];
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                    const int in_c = in_c_off + lane_idx;
                    if (in_c < dim.in.c) {
                        smem.sparse.s_in[element_idx][lane_idx] = batch_in[(in_y_im * in_row_vals + in_x_im * dim.in.c) + in_c];
                    } else {
                        smem.sparse.s_in[element_idx][lane_idx] = 0.0f;
                    }
                }

                __syncthreads();
                if (out_c < dim.out.c) {
                    const int n_t_f = 8;

                    #pragma unroll 1
                    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                        bool inside_y = false;
                        for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                            const int in_px_idx = set_elements[element_idx];
                            if (in_px_idx < 0) {
                                break;
                            }
                            
                            const int in_y = (in_px_idx / w_in);
                            const int out_y = (in_y + kernel_y) * DILATION_Y - 1 + tile_start_out_y;
                            if (out_y >= 0 && out_y < dim.out.h) {
                                inside_y = true;
                                break;
                            }
                        }
                        if (!inside_y) {
                            continue;
                        }

                        #pragma unroll 1
                        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                            bool inside_x = false;
                            for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                                const int in_px_idx = set_elements[element_idx];
                                if (in_px_idx < 0) {
                                    break;
                                }
                                int in_x = (in_px_idx % w_in);
                                const int out_x = (in_x + kernel_x) * DILATION_X - 1 + tile_start_out_x;
                                if (out_x >= 0 && out_x < dim.out.w) {
                                    inside_x = true;
                                    break;
                                }
                            }
                            if (!inside_x) {
                                continue;
                            }
                            // #pragma unroll 1
                            for (int t_f_iter = 0; t_f_iter < WARP_SIZE / n_t_f && in_c_off + t_f_iter*n_t_f < dim.in.c; ++t_f_iter) {
                                const scalar_t *in_c_filter = &filter[((in_c_off + t_f_iter*n_t_f) * 9 + (1-kernel_y)*3 + (1-kernel_x)) * dim.out.c + out_c];
                                
                                float t_f[n_t_f];
                                #pragma unroll
                                for (int i_t_f = 0; i_t_f < n_t_f; ++i_t_f) {
                                    if (i_t_f + t_f_iter * n_t_f < dim.in.c) 
                                    {
                                        t_f[i_t_f] = in_c_filter[i_t_f * 9 * dim.out.c];
                                    } else {
                                        t_f[i_t_f] = 0.0f;
                                    }
                                }
                                
                                #pragma unroll
                                for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                                    const bool valid = set_elements[element_idx] >= 0;
                                    if (valid) {
                                        const int in_px_idx = set_elements[element_idx];
                                        int in_y = (in_px_idx / w_in) - 1;
                                        int in_x = (in_px_idx % w_in) - 1;
                                        const int out_y = (in_y + kernel_y);
                                        const int out_x = (in_x + kernel_x);
                                        const bool inside = out_y >= 0 && out_y < pixelsPerBlockY && out_x >= 0 && out_x < pixelsPerBlockX; 
                                        const float* s_in_px = smem.sparse.s_in[element_idx];
                                        if (inside) {
                                            float result = 0.0f;
                                            #pragma unroll
                                            for (int in_c_shared = 0; in_c_shared < n_t_f; ++in_c_shared) {
                                                result += s_in_px[in_c_shared + t_f_iter*n_t_f] * t_f[in_c_shared];
                                            }
                                            smem.sparse.s_out[out_y * pixelsPerBlockX + out_x][threadIdx.x] += result;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (out_c < dim.out.c) {
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = smem.sparse.s_out[out_y*pixelsPerBlockY + out_x][threadIdx.x];
                        }
                    }
                }
            }
        }
    }
    else {
#ifdef ENABLE_METRICS
        if (threadIdx.x == 0 && blockIdx.z == 0) {
            atomicAdd(&d_metrics->n_tiles_dense_mode, 1);
        }
#endif
        for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE) {
            const int out_c = out_c_off + threadIdx.x;
            scalar_t t_out[n_pixels_out];
            const scalar_t t_bias = bias == nullptr || out_c >= dim.out.c ? 0.0f : bias[out_c];
            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                t_out[i] = t_bias;
            }

            for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE) {
                __syncthreads();
                // only used vector instructions when input is aligned
                if (dim.in.c % 4 == 0) {
                    for (int px_idx = warp_idx * 4 + sub_warp_idx; px_idx < n_in_px; px_idx += n_warps*4) {
                        const int in_y = px_idx / w_in; 
                        const int in_x = px_idx % w_in;
                        const int in_c = in_c_off + sub_warp_lane_idx * 4;
                        const bool valid = in_c < dim.in.c && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);
                        
                        if (valid) {
                            const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                            const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                            const float4 val = reinterpret_cast<const float4*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 4];
                            reinterpret_cast<float4*>(&smem.dense_s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = val;
                        } else {
                            smem.dense_s_in[in_y * w_in + in_x][sub_warp_lane_idx*4] = 0.0f;
                            smem.dense_s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 1] = 0.0f;
                            smem.dense_s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 2] = 0.0f;
                            smem.dense_s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 3] = 0.0f;
                        }
                    }
                } else if (dim.in.c == 3) {
                    for (int val_idx = threadIdx.x; val_idx < n_in_px * 3; val_idx += BLOCK_SIZE) {
                        const int px_idx = val_idx / 3;
                        const int in_c = val_idx % 3;
                        const int in_y = px_idx / w_in; 
                        const int in_x = px_idx % w_in;
                        const bool valid = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

                        if (valid) {
                            const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                            const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                            smem.dense_s_in[in_y * w_in + in_x][in_c] = batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c];
                        } else {
                            smem.dense_s_in[in_y * w_in + in_x][in_c] = 0.0f;
                        }
                    }
                } 
                else {
                    for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                        const int in_y = px_idx / w_in; 
                        const int in_x = px_idx % w_in;
                        const int in_c = in_c_off + lane_idx;
                        const bool valid = in_c < dim.in.c && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

                        if (valid) {
                            const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                            const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                            smem.dense_s_in[in_y * w_in + in_x][lane_idx] = batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c];
                        } else {
                            smem.dense_s_in[in_y * w_in + in_x][lane_idx] = 0.0f;
                        }
                        
                    }
                }
                __syncthreads();

                for(int in_c = 0; in_c < 32 && in_c + in_c_off < dim.in.c; ++in_c) {
                    const scalar_t *in_c_filter = &filter[(in_c_off+in_c) * 9 * dim.out.c + out_c];
                    scalar_t t_f[9];
                    #pragma unroll
                    for(int f_y = 0; f_y < 3; ++f_y) {
                        #pragma unroll
                        for(int f_x = 0; f_x < 3; ++f_x) { 
                            t_f[f_y*3 + f_x] = in_c_filter[((2-f_y) * 3 + 2 - f_x) * dim.out.c];
                        }
                    }
                    
                    if (out_c < dim.out.c) {
                        #pragma unroll
                        for (int in_y = -1; in_y < h_in -1; ++in_y) {
                            #pragma unroll
                            for (int in_x = -1; in_x < w_in -1; ++in_x) {
                                const scalar_t val = smem.dense_s_in[(in_y + 1) * w_in + (in_x + 1)][in_c];
                                const int min_f_y = -in_y;
                                const int min_f_x = -in_x;  
                                const int max_f_y = h_in - in_y - 3;
                                const int max_f_x = w_in - in_x - 3;
                                const int stride_off_y = (((1-in_y) % STRIDE) + STRIDE) % STRIDE;
                                const int stride_off_x = (((1-in_x) % STRIDE) + STRIDE) % STRIDE;

                                #pragma unroll
                                for (int f_y = Utils::constexpr_max(-1 + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(1, max_f_y); f_y += STRIDE) {
                                    #pragma unroll
                                    for (int f_x = Utils::constexpr_max(-1 + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(1, max_f_x); f_x += STRIDE) {
                                        t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] += val * t_f[(f_y+1)*3 + f_x+1];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (out_c < dim.out.c) {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = t_out[out_y*pixelsPerBlockX + out_x];
                        }
                    }
                }
            }
        }
    }
}


template<typename scalar_t = float, int KERNEL_SIZE=3, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void deltacnn_dw_conv_sp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);


    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int K_HALF = (KERNEL_SIZE-1) / 2;
    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2 * K_HALF;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2 * K_HALF;
    const int n_in_px = w_in * h_in;
    const int lane_idx = threadIdx.x % WARP_SIZE;

    __shared__ uint32_t s_mask[n_in_px];
    uint32_t density = 0;
    uint64_t t_mask = 0LLU;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, ENABLE_DILATION>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    for (int out_c = tile_start_z + threadIdx.x; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE) {
        scalar_t t_out[n_pixels_out];
        const scalar_t t_bias = bias == nullptr ? 0.0f : bias[out_c];
        #pragma unroll
        for (int i = 0; i < n_pixels_out; ++i) {
            t_out[i] = t_bias;
        }
        
        const scalar_t *in_c_filter = &filter[out_c];
        scalar_t t_f[KERNEL_SIZE*KERNEL_SIZE];
        #pragma unroll
        for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
            #pragma unroll
            for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[((2*K_HALF-f_y) * KERNEL_SIZE + 2*K_HALF - f_x) * dim.out.c];
            }
        }
        
        #pragma unroll
        for (int in_y = -K_HALF; in_y < w_in - K_HALF; ++in_y) {
            #pragma unroll
            for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                const bool valid = is_px_mask_set<w_in, n_in_px>(in_x+K_HALF, in_y+K_HALF, t_mask, s_mask); 
                
                if (valid) {
                    const int in_y_im = (in_y+K_HALF) * DILATION_Y + tile_start_in_y;
                    const int in_x_im = (in_x+K_HALF) * DILATION_X + tile_start_in_x;
                    const scalar_t val = batch_in[(in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c];

                    const int min_f_y = -in_y;
                    const int min_f_x = -in_x;
                    const int max_f_y = w_in - in_y - KERNEL_SIZE;
                    const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                    const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                    const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                    #pragma unroll
                    for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                        #pragma unroll
                        for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                            t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] += val * t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF];
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
            const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
            #pragma unroll
            for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                if (valid) {
                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = t_out[out_y*pixelsPerBlockX + out_x];
                }
            }
        }
    }
}




template<typename scalar_t = float, int KERNEL_SIZE=3, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void deltacnn_standard_conv_sp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);


    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int K_HALF = (KERNEL_SIZE-1) / 2;
    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2 * K_HALF;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2 * K_HALF;
    const int n_in_px = w_in * h_in;
    const int in_row_vals = dim.in.w * dim.in.c;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int sub_warp_idx = lane_idx / 8;
    const int sub_warp_lane_idx = lane_idx % 8;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ scalar_t s_in[n_in_px][WARP_SIZE];
    __shared__ uint32_t s_mask[n_in_px];
    uint32_t density = 0;
    uint64_t t_mask = 0LLU;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, ENABLE_DILATION>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    // TODO add sparse mode
    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE) {
        const int out_c = out_c_off + threadIdx.x;
        scalar_t t_out[n_pixels_out];
        const scalar_t t_bias = bias == nullptr || out_c >= dim.out.c ? 0.0f : bias[out_c];

        #pragma unroll
        for (int i = 0; i < n_pixels_out; ++i) {
            t_out[i] = t_bias;
        }

        for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE) {
            __syncthreads();
            // only used vector instructions when input is aligned
            if (dim.in.c % 4 == 0) {
                for (int px_idx = warp_idx * 4 + sub_warp_idx; px_idx < n_in_px; px_idx += n_warps*4) {
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const int in_c = in_c_off + sub_warp_lane_idx * 4;
                    const bool valid = in_c < dim.in.c && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);
                    
                    if (valid) {
                        const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                        const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                        const float4 val = reinterpret_cast<const float4*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 4];
                        reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = val;
                    } else {
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4] = 0.0f;
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 1] = 0.0f;
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 2] = 0.0f;
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 3] = 0.0f;
                    }
                }
            } else if (dim.in.c == 3) {
                for (int val_idx = threadIdx.x; val_idx < n_in_px * 3; val_idx += BLOCK_SIZE) {
                    const int px_idx = val_idx / 3;
                    const int in_c = val_idx % 3;
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const bool valid = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

                    if (valid) {
                        const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                        const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                        s_in[in_y * w_in + in_x][in_c] = batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c];
                    } else {
                        s_in[in_y * w_in + in_x][in_c] = 0.0f;
                    }
                }
            } 
            else {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const int in_c = in_c_off + lane_idx;
                    const bool valid = in_c < dim.in.c && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

                    if (valid) {
                        const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                        const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                        s_in[in_y * w_in + in_x][lane_idx] = batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c];
                    } else {
                        s_in[in_y * w_in + in_x][lane_idx] = 0.0f;
                    }
                    
                }
            }
            __syncthreads();
            

            if (out_c < dim.out.c) {
                for(int in_c = 0; in_c < 32 && in_c + in_c_off < dim.in.c; ++in_c) {
                    const scalar_t *in_c_filter = &filter[(in_c_off+in_c) * KERNEL_SIZE*KERNEL_SIZE * dim.out.c + out_c];
                    scalar_t t_f[KERNEL_SIZE*KERNEL_SIZE];
                    #pragma unroll
                    for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                        #pragma unroll
                        for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                            t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[((KERNEL_SIZE-1-f_y) * KERNEL_SIZE + (KERNEL_SIZE-1) - f_x) * dim.out.c];
                        }
                    }
                
                    #pragma unroll
                    for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                        #pragma unroll
                        for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                            const scalar_t val = s_in[(in_y+K_HALF) * w_in + (in_x+K_HALF)][in_c];

                            // TODO try to skip pixels where mask is not set -> might be worth it in this 7x7 kernel
                            const int min_f_y = -in_y;
                            const int min_f_x = -in_x;
                            const int max_f_y = h_in - in_y - KERNEL_SIZE;
                            const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                            const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                            const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                            #pragma unroll
                            for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                                #pragma unroll
                                for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] += val * t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF];
                                }
                            }
                        }
                    }
                }
            }
        }
        

        if (out_c < dim.out.c) {
            #pragma unroll
            for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                #pragma unroll
                for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                    const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                    const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                    if (valid) {
                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = t_out[out_y*pixelsPerBlockX + out_x];
                    }
                }
            }
        }
    }
}

template<typename scalar_t = half, int KERNEL_SIZE=3, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void deltacnn_dw_conv_hp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);


    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int K_HALF = (KERNEL_SIZE-1) / 2;
    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2 * K_HALF;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2 * K_HALF;
    const int n_in_px = w_in * h_in;
    const int lane_idx = threadIdx.x % WARP_SIZE;

    __shared__ uint32_t s_mask[n_in_px];
    uint32_t density = 0;
    uint64_t t_mask = 0LLU;    

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero_hp<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, ENABLE_DILATION>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    for (int out_c = tile_start_z + threadIdx.x * 2; out_c < dim.out.c && (FULL_DEPTH || out_c < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c += BLOCK_SIZE*2) {
        half2 t_out[n_pixels_out];

        half2 t_bias = __float2half2_rn(0.0f);
        if (out_c + 1 < dim.out.c && bias != nullptr) {
             t_bias = *reinterpret_cast<const half2*>(&bias[out_c]);
        } else if (bias != nullptr) {
             t_bias = __halves2half2(bias[out_c], __float2half(0.0f));
        }
        #pragma unroll
        for (int i = 0; i < n_pixels_out; ++i) {
            t_out[i] = t_bias;
        }
        
        const scalar_t *in_c_filter = &filter[out_c];
        half2 t_f[KERNEL_SIZE*KERNEL_SIZE];
        if (dim.out.c % 2 == 0) {
            #pragma unroll
            for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                #pragma unroll
                for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                    t_f[f_y*KERNEL_SIZE + f_x] = *reinterpret_cast<const half2*>(&in_c_filter[((KERNEL_SIZE-1-f_y) * KERNEL_SIZE + KERNEL_SIZE-1 - f_x) * dim.out.c]);
                }
            }
        } 
        else {
            if (out_c + 1 < dim.out.c) {
                #pragma unroll
                for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                    #pragma unroll
                    for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                        t_f[f_y*KERNEL_SIZE + f_x] = __halves2half2(in_c_filter[((KERNEL_SIZE-1-f_y) * KERNEL_SIZE + KERNEL_SIZE-1 - f_x) * dim.out.c], in_c_filter[((KERNEL_SIZE-1-f_y) * KERNEL_SIZE + KERNEL_SIZE-1 - f_x) * dim.out.c + 1]);
                    }
                }
            } else {
                #pragma unroll
                for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                    #pragma unroll
                    for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                        t_f[f_y*KERNEL_SIZE + f_x] = __halves2half2(in_c_filter[((KERNEL_SIZE-1-f_y) * KERNEL_SIZE + KERNEL_SIZE-1 - f_x) * dim.out.c], __float2half(0.0f));
                    }
                }
            }
        }
        
        if (dim.out.c % 2 == 0) {
            #pragma unroll
            for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                #pragma unroll
                for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                    const bool valid = is_px_mask_set<w_in, n_in_px>(in_x+K_HALF, in_y+K_HALF, t_mask, s_mask);
                    if (valid) {
                        const int in_y_im = (in_y+K_HALF) * DILATION_Y + tile_start_in_y;
                        const int in_x_im = (in_x+K_HALF) * DILATION_X + tile_start_in_x;
                        const half2 val = *reinterpret_cast<const half2*>(&batch_in[(in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c]);
                        const int min_f_y = -in_y;
                        const int min_f_x = -in_x;  
                        const int max_f_y = h_in - in_y - KERNEL_SIZE;
                        const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                        const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                        const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;

                        #pragma unroll
                        for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                            #pragma unroll
                            for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] = __hfma2(val, t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF], t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)]);
                            }
                        }
                    }
                }
            }
        } else {
            if (out_c + 1 < dim.out.c) {
                #pragma unroll
                for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                    #pragma unroll
                    for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                        const bool valid = is_px_mask_set<w_in, n_in_px>(in_x+K_HALF, in_y+K_HALF, t_mask, s_mask);
                        if (valid) {
                            const int in_y_im = (in_y+K_HALF) * DILATION_Y + tile_start_in_y;
                            const int in_x_im = (in_x+K_HALF) * DILATION_X + tile_start_in_x;
                            const half2 val = __halves2half2(
                                batch_in[(in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c],
                                batch_in[(in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c + 1]
                            );
                            const int min_f_y = -in_y;
                            const int min_f_x = -in_x;  
                            const int max_f_y = h_in - in_y - KERNEL_SIZE;
                            const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                            const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                            const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;

                            #pragma unroll
                            for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                                #pragma unroll
                                for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] = __hfma2(val, t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF], t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)]);
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                    #pragma unroll
                    for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                        const bool valid = is_px_mask_set<w_in, n_in_px>(in_x+K_HALF, in_y+K_HALF, t_mask, s_mask);
                        if (valid) {
                            const int in_y_im = (in_y+K_HALF) * DILATION_Y + tile_start_in_y;
                            const int in_x_im = (in_x+K_HALF) * DILATION_X + tile_start_in_x;
                            const half2 val = __halves2half2(
                                batch_in[(in_y_im * dim.in.w + in_x_im) * dim.in.c + out_c],
                                __float2half(0.0f)
                            );
                            
                            const int min_f_y = -in_y;
                            const int min_f_x = -in_x;  
                            const int max_f_y = h_in - in_y - KERNEL_SIZE;
                            const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                            const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                            const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;

                            #pragma unroll
                            for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                                #pragma unroll
                                for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] = __hfma2(val, t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF], t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)]);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (dim.out.c % 2 == 0) {
            #pragma unroll
            for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                #pragma unroll
                for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                    const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                    const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                    if (valid) {
                        *reinterpret_cast<half2*>(&batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c]) = t_out[out_y*pixelsPerBlockX + out_x];
                    }
                }
            }
        } else {
            if (out_c + 1 < dim.out.c) {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c + 1] = __high2half(t_out[out_y*pixelsPerBlockX + out_x]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                        }
                    }
                }
            }
        }
    }
}


template<typename scalar_t = half, int KERNEL_SIZE=3, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void deltacnn_standard_conv_hp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    const int OUT_CHANNELS_PER_THREAD = 2;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);


    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int K_HALF = (KERNEL_SIZE-1) / 2;
    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2 * K_HALF;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2 * K_HALF;
    const int n_in_px = w_in * h_in;
    const int in_row_vals = dim.in.w * dim.in.c;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    const uint16_t out_c_aligned = divup(dim.out.c, 2) * 2;
    const uint16_t in_c_aligned = divup(dim.in.c, 2) * 2;

    __shared__ half2 s_in[n_in_px][WARP_SIZE];
    __shared__ uint32_t s_mask[n_in_px];
    uint32_t density = 0;
    uint64_t t_mask = 0LLU;    

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);
    

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero_hp<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, ENABLE_DILATION>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }


    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE*OUT_CHANNELS_PER_THREAD) {
        const int out_c = out_c_off + threadIdx.x * 2;
        half2 t_out[n_pixels_out];

        const half2 t_bias = bias == nullptr || out_c >= dim.out.c ? __half2half2(__float2half(0.0f)) : *reinterpret_cast<const half2*>(&bias[out_c]);
        #pragma unroll
        for (int i = 0; i < n_pixels_out; ++i) {
            t_out[i] = t_bias;
        }

        for (int in_c_off = 0; in_c_off < in_c_aligned; in_c_off += WARP_SIZE * 2) {
            __syncthreads();
            if (dim.in.c % 2 == 0) {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                    const int in_c = in_c_off + lane_idx * 2;
                    const bool valid = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                    if (valid) {
                        half2 val = reinterpret_cast<const half2*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 2];
                        s_in[in_y * w_in + in_x][lane_idx] = val;
                    } else {
                        s_in[in_y * w_in + in_x][lane_idx] = __half2half2(__float2half(0.0f));
                    }
                }
            } else if (dim.in.c == 3) {
                for (int val_idx = threadIdx.x; val_idx < n_in_px * 2; val_idx += BLOCK_SIZE) {
                    const int px_idx = val_idx / 2;
                    const int in_c = in_c_off + (val_idx % 2) * 2;
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                    const bool valid1 = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                    const bool valid2 = in_c + 1 < dim.in.c;
                    
                    half2 val;
                    if (valid1 && valid2) {
                        val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                    }
                    else if (valid1) {
                        val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                    } else {
                        val = __half2half2(__float2half(0.0f));
                    }
                    s_in[in_y * w_in + in_x][(in_c-in_c_off)/2] = val;
                }
            } else {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                    const int in_c = in_c_off + lane_idx * 2;
                    const bool valid1 = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                    const bool valid2 = in_c + 1 < dim.in.c;
                    
                    half2 val;
                    if (valid1 && valid2) {
                        val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                    }
                    else if (valid1) {
                        val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                    } else {
                        val = __half2half2(__float2half(0.0f));
                    }
                    s_in[in_y * w_in + in_x][lane_idx] = val;
                }
            }
            __syncthreads();
            // aligned, because filters are always layed out as tuples
            for(int in_c = 0; in_c < 64 && in_c + in_c_off < in_c_aligned; ++in_c) {
                half2 t_f[KERNEL_SIZE*KERNEL_SIZE];

                const half *in_c_filter1 = &filter[(in_c_off+in_c) * KERNEL_SIZE*KERNEL_SIZE * out_c_aligned + out_c];
                #pragma unroll
                for(int f_px_idx = 0; f_px_idx < KERNEL_SIZE*KERNEL_SIZE; ++f_px_idx) {
                    t_f[(KERNEL_SIZE*KERNEL_SIZE-1)-f_px_idx] = *reinterpret_cast<const half2*>(&in_c_filter1[f_px_idx * out_c_aligned]);
                }

                if (out_c < dim.out.c) {
                    // in_c % 2 check is used to speedup half2 multiplications without increasing registers much.
                    // if == 0 --> take values as is and multiply them with filter
                    // if != 0 --> swap values and multiply them with filter next filter pair
                    // the filter pairs are pre-processed to match this pattern
                    if (in_c % 2 == 0) {
                        #pragma unroll
                        for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                            #pragma unroll
                            for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                                const half2 val = s_in[((in_y+K_HALF) * w_in + (in_x+K_HALF))][in_c/2];
                                const int min_f_y = -in_y;
                                const int min_f_x = -in_x;  
                                const int max_f_y = h_in - in_y - KERNEL_SIZE;
                                const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                                const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                                const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;

                                #pragma unroll
                                for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                                    #pragma unroll
                                    for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                        t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] = __hfma2(val, t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF], t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)]);
                                    }
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                            #pragma unroll
                            for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                                const half2 val = __lowhigh2highlow(s_in[((in_y+K_HALF) * w_in + (in_x+K_HALF))][in_c/2]);
                                const int min_f_y = -in_y;
                                const int min_f_x = -in_x;  
                                const int max_f_y = h_in - in_y - KERNEL_SIZE;
                                const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                                const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                                const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;

                                #pragma unroll
                                for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                                    #pragma unroll
                                    for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {
                                        t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] = __hfma2(val, t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF], t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        
        if (out_c + 1 < dim.out.c) {
            if (dim.out.c % 2 == 0) {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            *reinterpret_cast<half2*>(&batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c]) = t_out[out_y*pixelsPerBlockX + out_x];
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c + 1] = __high2half(t_out[out_y*pixelsPerBlockX + out_x]);
                        }
                    }
                }
            }
        }
        else if (out_c < dim.out.c) {
            #pragma unroll
            for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                #pragma unroll
                for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                    const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                    const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                    if (valid) {
                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                    }
                }
            }
        }
    }
}



template<typename scalar_t = half, int pixelsPerBlockX=6, int pixelsPerBlockY=6, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, int OUT_CHANNELS_PER_THREAD=2, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void 
deltacnn_3x3_hp(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ filter,
    const scalar_t* __restrict__ bias,
    const uint32_t* __restrict__ mask,
    uint32_t* __restrict__ out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    const int in_row_vals = dim.in.w * dim.in.c;
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int KERNEL_SIZE = 3;
    
    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION>(
        tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim
    );

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);

    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2;
    const int n_in_px = w_in * h_in;
    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    const int sparse_mode_max_elements = 4;

    struct SparseSMEM {
        half2 s_in[sparse_mode_max_elements][WARP_SIZE];
        half2 s_out[n_pixels_out][BLOCK_SIZE];
    };
    union SMEM {
        SparseSMEM sparse;
        half2 dense_s_in[n_in_px][WARP_SIZE];
    };

    __shared__ SMEM smem;
    __shared__ uint32_t s_mask[n_in_px];

    const uint16_t out_c_aligned = divup(dim.out.c, 2) * 2;
    const uint16_t in_c_aligned = divup(dim.in.c, 2) * 2;

    uint64_t t_mask = 0LLU;
    uint32_t density = 0;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, ENABLE_DILATION, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);


    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if ((n_in_px <= 64 && t_mask == 0LLU) || (n_in_px > 64 && density == 0)) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero_hp<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, ENABLE_DILATION>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }


    // TODO debug sparse kernel. disabled for now
    if (SUB_TILE_SPARSITY && density <= sparse_mode_max_elements && !ENABLE_DILATION && STRIDE == 1 && false) {
        // TODO implement dilated and strided sparse kernels
        int set_elements[sparse_mode_max_elements];
        int last_element_idx = -1;
        #pragma unroll
        for (int update_idx = 0; update_idx < sparse_mode_max_elements; ++update_idx) {
            set_elements[update_idx] = -1;
            for (int i = last_element_idx + 1; i < n_in_px; ++i) {
                if (s_mask[i] != 0) {
                    set_elements[update_idx] = i;
                    last_element_idx = i;
                    break;
                }
            }
        }

        for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE*OUT_CHANNELS_PER_THREAD) {
            const int out_c = out_c_off + threadIdx.x * 2;
            const half2 t_bias = bias == nullptr || out_c >= dim.out.c ? __half2half2(__float2half(0.0f)) : *reinterpret_cast<const half2*>(&bias[out_c]);
            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                smem.sparse.s_out[i][threadIdx.x] = t_bias;
            }

            for (int in_c_off = 0; in_c_off < in_c_aligned; in_c_off += WARP_SIZE * OUT_CHANNELS_PER_THREAD) {
                __syncthreads();
                if (dim.in.c % 2 == 0) {
                    for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                        if (set_elements[element_idx] == -1 || element_idx % n_warps != warp_idx) {
                            continue;
                        }
                        const int px_idx = set_elements[element_idx];
                        const int in_y = px_idx / w_in; 
                        const int in_y_im = in_y + tile_start_in_y;
                        const int in_x = px_idx % w_in;
                        const int in_x_im = in_x + tile_start_in_x; 
                        const int in_c = in_c_off + lane_idx * 2;
                        const bool valid = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                        if (valid) {
                            half2 val = reinterpret_cast<const half2*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 2];
                            smem.sparse.s_in[element_idx][lane_idx] = val;
                        } else {
                            smem.sparse.s_in[element_idx][lane_idx] = __half2half2(__float2half(0.0f));
                        }
                    }
                } else {
                    for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                        if (set_elements[element_idx] == -1 || element_idx % n_warps != warp_idx) {
                            continue;
                        }
                        const int px_idx = set_elements[element_idx];
                        const int in_y = px_idx / w_in; 
                        const int in_y_im = in_y + tile_start_in_y;
                        const int in_x = px_idx % w_in;
                        const int in_x_im = in_x + tile_start_in_x; 
                        const int in_c = in_c_off + lane_idx * 2;
                        const bool valid1 = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                        const bool valid2 = in_c + 1 < dim.in.c;
                        
                        half2 val;
                        if (valid1 && valid2) {
                            val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                        }
                        else if (valid1) {
                            val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                        } else {
                            val = __half2half2(__float2half(0.0f));
                        }
                        smem.sparse.s_in[element_idx][lane_idx] = val;
                    }
                }
                __syncthreads();
                if (out_c >= dim.out.c)
                    continue;

                const int n_t_f = 4;

                #pragma unroll 1
                for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                    bool inside_y = false;
                    for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                        const int in_px_idx = set_elements[element_idx];
                        if (in_px_idx < 0) {
                            break;
                        }
                        const int in_y = (in_px_idx / w_in) - DILATION_Y;
                        const int out_y = in_y + kernel_y * DILATION_Y;
                        if (out_y >= 0 && out_y < dim.out.h) {
                            inside_y = true;
                            break;
                        }
                    }
                    if (!inside_y) {
                        continue;
                    }
                    #pragma unroll 1
                    for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                        bool inside_x = false;
                        for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                            const int in_px_idx = set_elements[element_idx];
                            if (in_px_idx < 0) {
                                break;
                            }
                            int in_x = (in_px_idx % w_in) - DILATION_X;
                            const int out_x = in_x + kernel_x * DILATION_X;
                            if (out_x >= 0 && out_x < dim.out.w) {
                                inside_x = true;
                                break;
                            }
                        }
                        if (!inside_x) {
                            continue;
                        }

                        for (int t_f_iter = 0; t_f_iter < WARP_SIZE / n_t_f && in_c_off + t_f_iter*n_t_f < in_c_aligned; ++t_f_iter) {
                            const scalar_t *in_c_filter = &filter[((in_c_off + t_f_iter*n_t_f*2) * 9 + (1-kernel_y)*3 + (1-kernel_x)) * out_c_aligned + out_c];
                            
                            half2 t_f[n_t_f];
                            for (int in_c_iter = 0; in_c_iter < 2; ++in_c_iter) {
                                #pragma unroll
                                for (int i_t_f = 0; i_t_f < n_t_f; ++i_t_f) {
                                    if (i_t_f * 2 + t_f_iter * n_t_f * 2 < in_c_aligned)
                                    {
                                        t_f[i_t_f] = *reinterpret_cast<const half2*>(&in_c_filter[((in_c_iter + i_t_f * 2) * 9) * out_c_aligned]);
                                    } else {
                                        t_f[i_t_f] = __float2half2_rn(0.0f);
                                    }
                                }

                                if (in_c_iter == 0) {
                                    #pragma unroll
                                    for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                                        const bool valid = set_elements[element_idx] >= 0;
                                        if (valid) {
                                            const int in_px_idx = set_elements[element_idx];
                                            int in_y = (in_px_idx / w_in) - DILATION_Y;
                                            int in_x = (in_px_idx % w_in) - DILATION_X;
                                            const int out_y = in_y + kernel_y * DILATION_Y;
                                            const int out_x = in_x + kernel_x * DILATION_X;
                                            const bool inside = out_y >= 0 && out_y < pixelsPerBlockY && out_x >= 0 && out_x < pixelsPerBlockX; 
                                            const half2* s_in_px = smem.sparse.s_in[element_idx];
                                            if (inside) {
                                                half2 result = __float2half2_rn(0.0f);
                                                #pragma unroll
                                                for (int in_c_shared = 0; in_c_shared < n_t_f; ++in_c_shared) {
                                                    result = __hfma2(s_in_px[in_c_shared + t_f_iter*n_t_f], t_f[in_c_shared], result);
                                                }
                                                smem.sparse.s_out[out_y * pixelsPerBlockX + out_x][threadIdx.x] = __hadd2(result, smem.sparse.s_out[out_y * pixelsPerBlockX + out_x][threadIdx.x]);
                                            }
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int element_idx = 0; element_idx < sparse_mode_max_elements; ++element_idx) {
                                        const bool valid = set_elements[element_idx] >= 0;
                                        if (valid) {
                                            const int in_px_idx = set_elements[element_idx];
                                            int in_y = (in_px_idx / w_in) - DILATION_Y;
                                            int in_x = (in_px_idx % w_in) - DILATION_X;
                                            const int out_y = in_y + kernel_y * DILATION_Y;
                                            const int out_x = in_x + kernel_x * DILATION_X;
                                            const bool inside = out_y >= 0 && out_y < pixelsPerBlockY && out_x >= 0 && out_x < pixelsPerBlockX; 
                                            const half2* s_in_px = smem.sparse.s_in[element_idx];
                                            if (inside) {
                                                half2 result = __float2half2_rn(0.0f);
                                                #pragma unroll
                                                for (int in_c_shared = 0; in_c_shared < n_t_f; ++in_c_shared) {
                                                    result = __hfma2(__lowhigh2highlow(s_in_px[in_c_shared + t_f_iter*n_t_f]), t_f[in_c_shared], result);
                                                }
                                                smem.sparse.s_out[out_y * pixelsPerBlockX + out_x][threadIdx.x] = __hadd2(result, smem.sparse.s_out[out_y * pixelsPerBlockX + out_x][threadIdx.x]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (out_c + 1 < dim.out.c) {
                if (dim.out.c % 2 == 0) {
                    #pragma unroll
                    for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                        const int out_y_im = out_y + tile_start_out_y;
                        #pragma unroll
                        for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                            const int out_x_im = out_x + tile_start_out_x;
                            const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                            if (valid) {
                                reinterpret_cast<half2*>(batch_out)[((out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c) / 2] = smem.sparse.s_out[out_y*pixelsPerBlockX + out_x][threadIdx.x];
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                        const int out_y_im = out_y + tile_start_out_y;
                        #pragma unroll
                        for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                            const int out_x_im = out_x + tile_start_out_x;
                            const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                            if (valid) {
                                batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(smem.sparse.s_out[out_y*pixelsPerBlockX + out_x][threadIdx.x]);
                                batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c + 1] = __high2half(smem.sparse.s_out[out_y*pixelsPerBlockX + out_x][threadIdx.x]);
                            }
                        }
                    }
                }
            }
            else if (out_c < dim.out.c) {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(smem.sparse.s_out[out_y*pixelsPerBlockX + out_x][threadIdx.x]);
                        }
                    }
                }
            }
        }
    }
    else 
    {
        for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE*OUT_CHANNELS_PER_THREAD) {
            const int out_c = out_c_off + threadIdx.x * 2;
            half2 t_out[n_pixels_out];
            const half2 t_bias = bias == nullptr || out_c >= dim.out.c ? __half2half2(__float2half(0.0f)) : *reinterpret_cast<const half2*>(&bias[out_c]);
            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                t_out[i] = t_bias;
            }

            for (int in_c_off = 0; in_c_off < in_c_aligned; in_c_off += WARP_SIZE * OUT_CHANNELS_PER_THREAD) {
                __syncthreads();
                if (dim.in.c % 2 == 0) {
                    for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                        const int in_y = px_idx / w_in; 
                        const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                        const int in_x = px_idx % w_in;
                        const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                        const int in_c = in_c_off + lane_idx * 2;
                        const bool valid = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                        if (valid) {
                            half2 val = reinterpret_cast<const half2*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 2];
                            smem.dense_s_in[in_y * w_in + in_x][lane_idx] = val;
                        } else {
                            smem.dense_s_in[in_y * w_in + in_x][lane_idx] = __half2half2(__float2half(0.0f));
                        }
                    }
                } else if (dim.in.c == 3) {
                    for (int val_idx = threadIdx.x; val_idx < n_in_px * 2; val_idx += BLOCK_SIZE) {
                        const int px_idx = val_idx / 2;
                        const int in_c = (val_idx % 2) * 2;
                        const int in_y = px_idx / w_in; 
                        const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                        const int in_x = px_idx % w_in;
                        const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                        const bool valid1 = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                        const bool valid2 = in_c + 1 < dim.in.c;
                        
                        half2 val;
                        if (valid1 && valid2) {
                            val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                        }
                        else if (valid1) {
                            val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                        } else {
                            val = __half2half2(__float2half(0.0f));
                        }
                        smem.dense_s_in[in_y * w_in + in_x][in_c/2] = val;
                    }
                } else {
                    for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                        const int in_y = px_idx / w_in; 
                        const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                        const int in_x = px_idx % w_in;
                        const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                        const int in_c = in_c_off + lane_idx * 2;
                        const bool valid1 = is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                        const bool valid2 = in_c + 1 < dim.in.c;
                        
                        half2 val;
                        if (valid1 && valid2) {
                            val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                        }
                        else if (valid1) {
                            val = make_half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                        } else {
                            val = __half2half2(__float2half(0.0f));
                        }
                        smem.dense_s_in[in_y * w_in + in_x][lane_idx] = val;
                    }
                }
                __syncthreads();

                // aligned, because filters are always layed out as tuples
                for(int in_c = 0; in_c < 64 && in_c + in_c_off < in_c_aligned; ++in_c) {
                    half2 t_f[9];

                    const half *in_c_filter1 = &filter[(in_c_off+in_c) * 9 * out_c_aligned + out_c];
                    #pragma unroll
                    for(int f_px_idx = 0; f_px_idx < 9; ++f_px_idx) {
                        t_f[8-f_px_idx] = *reinterpret_cast<const half2*>(&in_c_filter1[f_px_idx * out_c_aligned]);
                    }

                    if (out_c < dim.out.c) {
                        if (in_c % 2 == 0) {
                            #pragma unroll
                            for (int in_y = -1; in_y < h_in-1; ++in_y) {
                                #pragma unroll
                                for (int in_x = -1; in_x < w_in-1; ++in_x) {
                                    half2 val = smem.dense_s_in[(in_y+1) * w_in + (in_x+1)][in_c / 2];
                                    const int min_f_y = -in_y;
                                    const int min_f_x = -in_x;  
                                    const int max_f_y = h_in - in_y - 3;
                                    const int max_f_x = w_in - in_x - 3;
                                    const int stride_off_y = (((1-in_y) % STRIDE) + STRIDE) % STRIDE;
                                    const int stride_off_x = (((1-in_x) % STRIDE) + STRIDE) % STRIDE;

                                    #pragma unroll
                                    for (int f_y = Utils::constexpr_max(-1 + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(1, max_f_y); f_y += STRIDE) {
                                        #pragma unroll
                                        for (int f_x = Utils::constexpr_max(-1 + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(1, max_f_x); f_x += STRIDE) {
                                            const int out_idx = ((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE);
                                            t_out[out_idx] = __hfma2(val, t_f[(f_y+1)*3 + (f_x+1)], t_out[out_idx]);
                                        }
                                    }
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int in_y = -1; in_y < h_in-1; ++in_y) {
                                #pragma unroll
                                for (int in_x = -1; in_x < w_in-1; ++in_x) {
                                    half2 val = __lowhigh2highlow(smem.dense_s_in[(in_y+1) * w_in + (in_x+1)][in_c / 2]);
                                    const int min_f_y = -in_y;
                                    const int min_f_x = -in_x;  
                                    const int max_f_y = h_in - in_y - 3;
                                    const int max_f_x = w_in - in_x - 3;
                                    const int stride_off_y = (((1-in_y) % STRIDE) + STRIDE) % STRIDE;
                                    const int stride_off_x = (((1-in_x) % STRIDE) + STRIDE) % STRIDE;

                                    #pragma unroll
                                    for (int f_y = Utils::constexpr_max(-1 + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(1, max_f_y); f_y += STRIDE) {
                                        #pragma unroll
                                        for (int f_x = Utils::constexpr_max(-1 + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(1, max_f_x); f_x += STRIDE) {
                                            const int out_idx = ((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE);
                                            t_out[out_idx] = __hfma2(val, t_f[(f_y+1)*3 + (f_x+1)], t_out[out_idx]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (out_c + 1 < dim.out.c) {
                if (dim.out.c % 2 == 0) {
                    #pragma unroll
                    for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                        const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                        #pragma unroll
                        for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                            const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                            const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                            if (valid) {
                                reinterpret_cast<float*>(batch_out)[((out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c) / 2] = reinterpret_cast<float*>(t_out)[out_y*pixelsPerBlockX + out_x];
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                        const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                        #pragma unroll
                        for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                            const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                            const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                            if (valid) {
                                batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                                batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c + 1] = __high2half(t_out[out_y*pixelsPerBlockX + out_x]);
                            }
                        }
                    }
                }
            }
            else if (out_c < dim.out.c) {
                #pragma unroll
                for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                    const int out_y_im = out_y + tile_start_out_y;
                    #pragma unroll
                    for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                        const int out_x_im = out_x + tile_start_out_x;
                        const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                        if (valid) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                        }
                    }
                }
            }
        }
    }
}


template<typename scalar_t = float, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, int OUT_C_PER_THREAD=8, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1>
__global__ void deltacnn_1x1_sp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int in_row_vals = dim.in.w * dim.in.c;
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int KERNEL_SIZE = 1;

    
    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, false>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);

    const uint8_t w_in = pixelsPerBlockX;
    const uint8_t n_in_px = pixelsPerBlockX * pixelsPerBlockY;
    const uint8_t n_in_px_aligned = divup(n_in_px, 4) * 4;

    const uint8_t lane_idx = threadIdx.x % WARP_SIZE;
    const uint8_t warp_idx = threadIdx.x / WARP_SIZE;
    const uint8_t sub_warp_idx = lane_idx / 8;
    const uint8_t sub_warp_lane_idx = lane_idx % 8;
    const uint8_t n_warps = BLOCK_SIZE / WARP_SIZE;

    const uint8_t s_in_channels = 16;
    const uint8_t out_px_per_thread = divup(n_pixels_out, n_warps);

    __shared__ scalar_t s_in[n_in_px_aligned][WARP_SIZE];
     uint32_t* s_mask = reinterpret_cast<uint32_t*>(s_in);
    __shared__ scalar_t s_f[s_in_channels][OUT_C_PER_THREAD * WARP_SIZE];
    uint64_t t_mask = 0LLU;
    uint32_t density;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, false, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, false, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, false>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    // TODO implement a special sparse inference mode --> current sub_tile_sparse mode is very expensive

    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += OUT_C_PER_THREAD * WARP_SIZE) {
        scalar_t t_out[out_px_per_thread][OUT_C_PER_THREAD];
        #pragma unroll
        for (int out_c_idx = 0; out_c_idx < OUT_C_PER_THREAD/4; ++out_c_idx) {
            const int out_c_gl = out_c_off + out_c_idx * WARP_SIZE * 4 + lane_idx * 4;
            #pragma unroll
            for (int out_c_iter = 0; out_c_iter < 4; ++out_c_iter) {
                if (out_c_gl + out_c_iter < dim.out.c) {
                    const scalar_t t_bias = bias == nullptr ? 0.0f : bias[out_c_gl + out_c_iter];
                    #pragma unroll
                    for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        t_out[px_idx][out_c_idx*4 + out_c_iter] = t_bias;
                    }
                }
            }
        }

        // sync here to make sure that no thread is accessing the shared mask anymore
        __syncthreads();
        for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE) {
            // load inputs
            // only use vector instructions when input is aligned
            if (dim.in.c % 4 == 0) {
                for (int px_idx = warp_idx * 4 + sub_warp_idx; px_idx < n_in_px; px_idx += n_warps*4) {
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * STRIDE + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * STRIDE + tile_start_in_x; 
                    const int in_c = in_c_off + sub_warp_lane_idx * 4;
                    const bool valid = ((t_mask & (1LLU << (in_y * w_in + in_x))) != 0) && in_c < dim.in.c;
                    
                    if (valid) {
                        const float4 val = reinterpret_cast<const float4*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 4];
                        reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = val;
                    } else {
                        reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            } 
            else {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * STRIDE + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * STRIDE + tile_start_in_x; 
                    const int in_c = in_c_off + lane_idx;
                    const bool valid = ((t_mask & (1LLU << (in_y * w_in + in_x))) != 0) && in_c < dim.in.c;
                    
                    s_in[in_y * w_in + in_x][lane_idx] = valid ? batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c] : 0.0f;
                }
            }

            const int in_c_per_iter = SUB_TILE_SPARSITY ? 4 : 8;
            for(int in_c = 0; in_c < 32 && in_c + in_c_off < dim.in.c; in_c += in_c_per_iter) {
                if (in_c % s_in_channels == 0) {
                    __syncthreads();
                    // load filters
                    const bool requires_checks = (s_in_channels + in_c_off + in_c) > dim.in.c || (OUT_C_PER_THREAD * WARP_SIZE + out_c_off) > dim.out.c || (dim.out.c % 4) != 0;
                    if (requires_checks)
                    {
                        for (int in_c_idx = warp_idx; in_c_idx < s_in_channels; in_c_idx += n_warps) {
                            for (int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD; ++out_c_iter) {
                                int in_c_global = in_c_idx + in_c_off + in_c;
                                int out_c_global = out_c_off + out_c_iter * WARP_SIZE + lane_idx;
                                const bool valid = in_c_global < dim.in.c && out_c_global < dim.out.c;
                                s_f[in_c_idx][out_c_iter * WARP_SIZE + lane_idx] = valid ? filter[in_c_global * dim.out.c + out_c_global] : 0.0f;
                            }
                        }
                    } 
                    else {
                        for (int in_c_idx = warp_idx; in_c_idx < s_in_channels; in_c_idx += n_warps) {
                            for (int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD/4; ++out_c_iter) {
                                int in_c_global = in_c_idx + in_c_off + in_c;
                                int out_c_local = (out_c_iter * WARP_SIZE + lane_idx) * 4;
                                *reinterpret_cast<float4*>(&s_f[in_c_idx][out_c_local]) = *reinterpret_cast<const float4*>(&filter[in_c_global * dim.out.c + out_c_local + out_c_off]);
                            }
                        }
                    }

                    __syncthreads();
                }

                
                if (out_c_off + lane_idx*4 < dim.out.c) {
                    float4 t_f[in_c_per_iter][OUT_C_PER_THREAD/4];
                    #pragma unroll
                    for (int in_c_iter = 0; in_c_iter < in_c_per_iter; ++in_c_iter) {
                        #pragma unroll
                        for (int t_out_c = 0; t_out_c < OUT_C_PER_THREAD / 4; ++t_out_c) {
                            t_f[in_c_iter][t_out_c] = *reinterpret_cast<float4*>(&s_f[(in_c % s_in_channels) + in_c_iter][t_out_c*4*WARP_SIZE + lane_idx * 4]);
                        }
                    }

                    #pragma unroll
                    for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        const bool valid = (t_mask & 1LLU << (px_idx * n_warps + warp_idx)) != 0;
                        if(!SUB_TILE_SPARSITY || valid) {
                            #pragma unroll
                            for (int in_c_iter=0; in_c_iter < in_c_per_iter; in_c_iter += 4) {
                                const float4 val = *reinterpret_cast<float4*>(&s_in[px_idx * n_warps + warp_idx][in_c + in_c_iter]);
                                #pragma unroll
                                for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 4; ++out_c_iter) {
                                    t_out[px_idx][out_c_iter*4]   += val.x * t_f[in_c_iter][out_c_iter].x;
                                    t_out[px_idx][out_c_iter*4+1] += val.x * t_f[in_c_iter][out_c_iter].y;
                                    t_out[px_idx][out_c_iter*4+2] += val.x * t_f[in_c_iter][out_c_iter].z;
                                    t_out[px_idx][out_c_iter*4+3] += val.x * t_f[in_c_iter][out_c_iter].w;
                                }
                                #pragma unroll
                                for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 4; ++out_c_iter) {
                                    t_out[px_idx][out_c_iter*4]   += val.y * t_f[in_c_iter+1][out_c_iter].x;
                                    t_out[px_idx][out_c_iter*4+1] += val.y * t_f[in_c_iter+1][out_c_iter].y;
                                    t_out[px_idx][out_c_iter*4+2] += val.y * t_f[in_c_iter+1][out_c_iter].z;
                                    t_out[px_idx][out_c_iter*4+3] += val.y * t_f[in_c_iter+1][out_c_iter].w;
                                }
                                #pragma unroll
                                for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 4; ++out_c_iter) {
                                    t_out[px_idx][out_c_iter*4]   += val.z * t_f[in_c_iter+2][out_c_iter].x;
                                    t_out[px_idx][out_c_iter*4+1] += val.z * t_f[in_c_iter+2][out_c_iter].y;
                                    t_out[px_idx][out_c_iter*4+2] += val.z * t_f[in_c_iter+2][out_c_iter].z;
                                    t_out[px_idx][out_c_iter*4+3] += val.z * t_f[in_c_iter+2][out_c_iter].w;
                                }
                                #pragma unroll
                                for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 4; ++out_c_iter) {
                                    t_out[px_idx][out_c_iter*4]   += val.w * t_f[in_c_iter+3][out_c_iter].x;
                                    t_out[px_idx][out_c_iter*4+1] += val.w * t_f[in_c_iter+3][out_c_iter].y;
                                    t_out[px_idx][out_c_iter*4+2] += val.w * t_f[in_c_iter+3][out_c_iter].z;
                                    t_out[px_idx][out_c_iter*4+3] += val.w * t_f[in_c_iter+3][out_c_iter].w;
                                }
                            }
                        }
                    }
                } 
                __syncthreads();
            }
        }

        if (out_c_off + lane_idx*4 < dim.out.c) {
            #pragma unroll
            for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                const int gl_px_idx = px_idx * n_warps + warp_idx;
                const int out_y = gl_px_idx / w_in;
                const int out_x = gl_px_idx % w_in;
                const int out_y_im = out_y + tile_start_out_y;
                const int out_x_im = out_x + tile_start_out_x;
                const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                if (valid) {
                    #pragma unroll
                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD/4; ++out_c_iter) {
                        const int out_c_gl = out_c_iter * 4 * WARP_SIZE + lane_idx * 4 + out_c_off;
                        // for some reason, uncoalesced write seems to be faster here
                        // if (out_c_gl + 3 < dim.out.c && dim.out.c % 4 == 0) {
                        //     float4 val = make_float4(t_out[px_idx][out_c_iter*4], t_out[px_idx][out_c_iter*4 + 1], t_out[px_idx][out_c_iter*4 + 2] ,t_out[px_idx][out_c_iter*4 + 3]);
                        //     *reinterpret_cast<float4*>(&batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl]) = val;
                        // } 
                        // else 
                        if (out_c_gl < dim.out.c) {
                            for (int out_c_sub_iter = 0; out_c_sub_iter < 4; ++out_c_sub_iter) {
                                if (out_c_gl + out_c_sub_iter < dim.out.c) {
                                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl + out_c_sub_iter] = t_out[px_idx][out_c_iter*4 + out_c_sub_iter];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename scalar_t = float, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, int OUT_C_PER_THREAD=8, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1>
__global__ void deltacnn_1x1_sp_2(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int in_row_vals = dim.in.w * dim.in.c;
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int KERNEL_SIZE = 1;

    
    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, false>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);

    const uint8_t w_in = pixelsPerBlockX;
    const uint8_t n_in_px = pixelsPerBlockX * pixelsPerBlockY;
    const uint8_t n_in_px_aligned = divup(n_in_px, 4) * 4;

    const uint8_t lane_idx = threadIdx.x % WARP_SIZE;
    const uint8_t warp_idx = threadIdx.x / WARP_SIZE;
    const uint8_t sub_warp_idx = lane_idx / 8;
    const uint8_t sub_warp_lane_idx = lane_idx % 8;
    const uint8_t n_warps = BLOCK_SIZE / WARP_SIZE;

    const uint8_t s_in_channels = 16;
    const uint8_t out_px_per_thread = divup(n_pixels_out, n_warps);

    __shared__ scalar_t s_in[n_in_px_aligned][WARP_SIZE];
     uint32_t* s_mask = reinterpret_cast<uint32_t*>(s_in);
    __shared__ scalar_t s_f[s_in_channels][OUT_C_PER_THREAD * WARP_SIZE];
    uint64_t t_mask = 0LLU;
    uint32_t density;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, false, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, false, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, false>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    // TODO implement a special sparse inference mode --> current sub_tile_sparse mode is very expensive

    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += OUT_C_PER_THREAD * WARP_SIZE) {
        scalar_t t_out[out_px_per_thread][OUT_C_PER_THREAD];

        #pragma unroll
        for (int out_c_idx = 0; out_c_idx < OUT_C_PER_THREAD; ++out_c_idx) {
            const int out_c_gl = out_c_off + out_c_idx * WARP_SIZE + lane_idx;
            if (out_c_gl < dim.out.c) {
                const scalar_t t_bias = bias == nullptr ? 0.0f : bias[out_c_gl];
                #pragma unroll
                for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                    t_out[px_idx][out_c_idx] = t_bias;
                }
            }
        }

        // sync here to make sure that no thread is accessing the shared mask anymore
        __syncthreads();
        for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE) {
            // load inputs
            // only use vector instructions when input is aligned
            if (dim.in.c % 4 == 0) {
                for (int px_idx = warp_idx * 4 + sub_warp_idx; px_idx < n_in_px; px_idx += n_warps*4) {
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * STRIDE + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * STRIDE + tile_start_in_x; 
                    const int in_c = in_c_off + sub_warp_lane_idx * 4;
                    const bool valid = ((t_mask & (1LLU << (in_y * w_in + in_x))) != 0) && in_c < dim.in.c;
                    
                    if (valid) {
                        const float4 val = reinterpret_cast<const float4*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 4];
                        reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = val;
                    } else {
                        reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            } 
            else {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * STRIDE + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * STRIDE + tile_start_in_x; 
                    const int in_c = in_c_off + lane_idx;
                    const bool valid = ((t_mask & (1LLU << (in_y * w_in + in_x))) != 0) && in_c < dim.in.c;
                    
                    s_in[in_y * w_in + in_x][lane_idx] = valid ? batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c] : 0.0f;
                }
            }

            const int in_c_per_iter = SUB_TILE_SPARSITY ? 4 : 8;
            for(int in_c = 0; in_c < 32 && in_c + in_c_off < dim.in.c; in_c += in_c_per_iter) {
                if (in_c % s_in_channels == 0) {
                    __syncthreads();
                    // load filters
                    const bool requires_checks = (s_in_channels + in_c_off + in_c) > dim.in.c || (OUT_C_PER_THREAD * WARP_SIZE + out_c_off) > dim.out.c;
                    if (requires_checks)
                    {
                        for (int in_c_idx = warp_idx; in_c_idx < s_in_channels; in_c_idx += n_warps) {
                            for (int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD; ++out_c_iter) {
                                int in_c_global = in_c_idx + in_c_off + in_c;
                                int out_c_global = out_c_off + out_c_iter * WARP_SIZE + lane_idx;
                                const bool valid = in_c_global < dim.in.c && out_c_global < dim.out.c;
                                s_f[in_c_idx][out_c_iter * WARP_SIZE + lane_idx] = valid ? filter[in_c_global * dim.out.c + out_c_global] : 0.0f;
                            }
                        }
                    } 
                    else {
                        for (int in_c_idx = warp_idx; in_c_idx < s_in_channels; in_c_idx += n_warps) {
                            for (int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD; ++out_c_iter) {
                                int in_c_global = in_c_idx + in_c_off + in_c;
                                int out_c_global = out_c_off + out_c_iter * WARP_SIZE + lane_idx;
                                s_f[in_c_idx][out_c_iter * WARP_SIZE + lane_idx] = filter[in_c_global * dim.out.c + out_c_global];
                            }
                        }
                    }

                    __syncthreads();
                }

                
                if (out_c_off + lane_idx < dim.out.c) {
                    float t_f[in_c_per_iter][OUT_C_PER_THREAD];
                    #pragma unroll
                    for (int in_c_iter = 0; in_c_iter < in_c_per_iter; ++in_c_iter) {
                        #pragma unroll
                        for (int t_out_c = 0; t_out_c < OUT_C_PER_THREAD; ++t_out_c) {
                            t_f[in_c_iter][t_out_c] = s_f[(in_c % s_in_channels) + in_c_iter][t_out_c*WARP_SIZE + lane_idx];
                        }
                    }

                    #pragma unroll
                    for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        const bool valid = (t_mask & 1LLU << (px_idx * n_warps + warp_idx)) != 0;
                        if(!SUB_TILE_SPARSITY || valid) {
                            #pragma unroll
                            for (int in_c_iter=0; in_c_iter < in_c_per_iter; in_c_iter += 4) {
                                const float4 val = *reinterpret_cast<float4*>(&s_in[px_idx * n_warps + warp_idx][in_c + in_c_iter]);
                                #pragma unroll
                                for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD; ++out_c_iter) {
                                    t_out[px_idx][out_c_iter]   += val.x * t_f[in_c_iter][out_c_iter];
                                    t_out[px_idx][out_c_iter]   += val.y * t_f[in_c_iter+1][out_c_iter];
                                    t_out[px_idx][out_c_iter]   += val.z * t_f[in_c_iter+2][out_c_iter];
                                    t_out[px_idx][out_c_iter]   += val.w * t_f[in_c_iter+3][out_c_iter];
                                }
                            }
                        }
                    }
                } 
                __syncthreads();
            }
        }

        if (out_c_off + lane_idx < dim.out.c) {
            #pragma unroll
            for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                const int gl_px_idx = px_idx * n_warps + warp_idx;
                const int out_y = gl_px_idx / w_in;
                const int out_x = gl_px_idx % w_in;
                const int out_y_im = out_y + tile_start_out_y;
                const int out_x_im = out_x + tile_start_out_x;
                const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                if (valid) {
                    #pragma unroll
                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD; ++out_c_iter) {
                        const int out_c_gl = out_c_iter * WARP_SIZE + lane_idx + out_c_off;
                        
                        if (out_c_gl < dim.out.c) {
                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl] = t_out[px_idx][out_c_iter];
                        }
                    }
                }
            }
        }
    }
}

template<typename scalar_t = half, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, int OUT_C_PER_THREAD=8, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1>
__global__ void deltacnn_1x1_hp(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int KERNEL_SIZE = 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, false>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);

    const uint8_t w_in = pixelsPerBlockX;
    const uint8_t n_in_px = pixelsPerBlockX * pixelsPerBlockY;
    const uint8_t n_in_px_aligned = divup(n_in_px, 4) * 4;
    const int in_row_vals = dim.in.w * dim.in.c;

    const uint8_t lane_idx = threadIdx.x % WARP_SIZE;
    const uint8_t warp_idx = threadIdx.x / WARP_SIZE;
    const uint8_t n_warps = BLOCK_SIZE / WARP_SIZE;

    const uint8_t s_in_channels = 16;
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const uint8_t out_px_per_thread = divup(n_pixels_out, n_warps);

    const uint16_t out_c_aligned = divup(dim.out.c,2)*2;

    __shared__ half2 s_in[n_in_px_aligned][WARP_SIZE];
     uint32_t* s_mask = reinterpret_cast<uint32_t*>(s_in);
    __shared__ half2 s_f[s_in_channels][OUT_C_PER_THREAD * WARP_SIZE];
    uint64_t t_mask = 0LLU;
    uint32_t density;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, false, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, false, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero_hp<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, false>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += OUT_C_PER_THREAD * WARP_SIZE) {
        half2 t_out[out_px_per_thread][OUT_C_PER_THREAD/2];
        if (dim.out.c % 2 == 0) {
            #pragma unroll
            for (int out_c_idx = 0; out_c_idx < OUT_C_PER_THREAD/8; ++out_c_idx) {
                const int out_c_gl = out_c_off + out_c_idx * WARP_SIZE * 8 + lane_idx * 8;
                #pragma unroll
                for (int out_c_iter = 0; out_c_iter < 4; ++out_c_iter) {
                    if (out_c_gl + out_c_iter < dim.out.c) {
                        if (out_c_gl + out_c_iter + 1 < dim.out.c) {
                            const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : *reinterpret_cast<const half2*>(&bias[out_c_gl + out_c_iter*2]);
                            #pragma unroll
                            for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                                t_out[px_idx][out_c_idx*4 + out_c_iter] = t_bias;
                            }
                        } else {
                            const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : __halves2half2(bias[out_c_gl + out_c_iter*2], __float2half(0.0f));
                            #pragma unroll
                            for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                                t_out[px_idx][out_c_idx*4 + out_c_iter] = t_bias;
                            }
                        }
                    }
                }
            }
        } else 
        {
            #pragma unroll
            for (int out_c_idx = 0; out_c_idx < OUT_C_PER_THREAD/8; ++out_c_idx) {
                const int out_c_gl = out_c_off + out_c_idx * WARP_SIZE * 8 + lane_idx * 8;
                #pragma unroll
                for (int out_c_iter = 0; out_c_iter < 4; ++out_c_iter) {
                    if (out_c_gl + out_c_iter < dim.out.c) {
                        if (out_c_gl + out_c_iter + 1 < dim.out.c) {
                            const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : __halves2half2(bias[out_c_gl + out_c_iter*2], bias[out_c_gl + out_c_iter*2 + 1]);
                            #pragma unroll
                            for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                                t_out[px_idx][out_c_idx*4 + out_c_iter] = t_bias;
                            }
                        } else {
                            const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : __halves2half2(bias[out_c_gl + out_c_iter*2], __float2half(0.0f));
                            #pragma unroll
                            for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                                t_out[px_idx][out_c_idx*4 + out_c_iter] = t_bias;
                            }
                        }
                    }
                }
            }
        }

        for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE*2) {
            // load inputs            
            __syncthreads();
            for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                const int in_y = px_idx / w_in; 
                const int in_y_im = in_y * STRIDE + tile_start_in_y;
                const int in_x = px_idx % w_in;
                const int in_x_im = in_x * STRIDE + tile_start_in_x; 
                const int in_c = in_c_off + lane_idx * 2;
                const bool valid = (t_mask & (1LLU << (in_y * w_in + in_x))) && in_c < dim.in.c;
                
                if (!valid) {
                    s_in[in_y * w_in + in_x][lane_idx] = __half2half2(__float2half(0.0f));
                }
                else {
                    if (in_c + 1 < dim.in.c) {
                        if (dim.in.c % 2 == 0) {
                            s_in[in_y * w_in + in_x][lane_idx] =
                                *reinterpret_cast<const half2*>(&batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c]);
                        } else {
                            s_in[in_y * w_in + in_x][lane_idx] = 
                                __halves2half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                        }
                    } else {
                        s_in[in_y * w_in + in_x][lane_idx] =
                            __halves2half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                    }
                }
            }

            // const int in_c_per_iter = SUB_TILE_SPARSITY ? 4 : 16;
            const int in_c_per_iter = SUB_TILE_SPARSITY ? 8 : 32;
            for(int in_c = 0; in_c < WARP_SIZE * 2 && in_c + in_c_off < dim.in.c; in_c += in_c_per_iter) {
                if (in_c % s_in_channels == 0) {
                    __syncthreads();
                    // load filters
                    // const bool requires_checks = (s_in_channels + in_c_off + in_c) > dim.in.c || (OUT_C_PER_THREAD * WARP_SIZE + out_c_off) > dim.out.c || (dim.out.c % 4) != 0;
                    const bool requires_checks = true;
                    if (requires_checks)
                    {
                        for (int in_c_idx = warp_idx*2; in_c_idx < s_in_channels; in_c_idx += n_warps*2) {
                            for (int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD/2; ++out_c_iter) {
                                int in_c_global = in_c_idx + in_c_off + in_c;
                                int out_c_global = out_c_off + out_c_iter * WARP_SIZE * 2 + lane_idx * 2;
                                const bool valid = in_c_global < dim.in.c && out_c_global < dim.out.c;
                                
                                s_f[in_c_idx][out_c_iter * WARP_SIZE + lane_idx] = valid ? 
                                    *reinterpret_cast<const half2*>(&filter[in_c_global * out_c_aligned + out_c_global]) : 
                                    __half2half2(__float2half(0.0f));
                                s_f[in_c_idx+1][out_c_iter * WARP_SIZE + lane_idx] = valid ? 
                                    *reinterpret_cast<const half2*>(&filter[(in_c_global+1) * out_c_aligned + out_c_global]) : 
                                    __half2half2(__float2half(0.0f));
                            }
                        }
                    }
                    __syncthreads();
                }

                
                if (out_c_off + lane_idx*8 < dim.out.c) {
                    half2 t_f[in_c_per_iter][OUT_C_PER_THREAD/2];

                    #pragma unroll
                    for (int in_c_iter = 0; in_c_iter < in_c_per_iter; ++in_c_iter) {
                        #pragma unroll
                        for (int t_out_c = 0; t_out_c < OUT_C_PER_THREAD / 2; t_out_c += 4) {
                            float4 f_temp = *reinterpret_cast<float4*>(&s_f[(in_c % s_in_channels) + in_c_iter][t_out_c*WARP_SIZE*4 + lane_idx * 4]);
                            t_f[in_c_iter][t_out_c] = *reinterpret_cast<half2*>(&f_temp.x);
                            t_f[in_c_iter][t_out_c+1] = *reinterpret_cast<half2*>(&f_temp.y);
                            t_f[in_c_iter][t_out_c+2] = *reinterpret_cast<half2*>(&f_temp.z);
                            t_f[in_c_iter][t_out_c+3] = *reinterpret_cast<half2*>(&f_temp.w);
                        }
                    }

                    // TODO vectorize loads -> save values and filters as float2 or float4, convert them on the fly to half2

                    #pragma unroll
                    for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        // const bool valid = (t_mask & 1LLU << (px_idx * n_warps + warp_idx)) != 0;
                        const bool valid = true;
                        if(!SUB_TILE_SPARSITY || valid) {
                            #pragma unroll
                            for (int in_c_iter=0; in_c_iter < in_c_per_iter; in_c_iter += 8) {
                                const float4 vals4 = *reinterpret_cast<float4*>(&s_in[px_idx * n_warps + warp_idx][(in_c + in_c_iter) / 2]);

                                {
                                    const half2 val = *reinterpret_cast<const half2*>(&vals4.x);
                                    const half2 val2 = __lowhigh2highlow(val);
                                    #pragma unroll
                                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 2; ++out_c_iter) {
                                        t_out[px_idx][out_c_iter] = __hfma2(val ,  t_f[in_c_iter][out_c_iter], t_out[px_idx][out_c_iter]);
                                        t_out[px_idx][out_c_iter] = __hfma2(val2 , t_f[in_c_iter+1][out_c_iter], t_out[px_idx][out_c_iter]);
                                    }
                                }

                                {
                                    const half2 val = *reinterpret_cast<const half2*>(&vals4.y);
                                    const half2 val2 = __lowhigh2highlow(val);
                                    #pragma unroll
                                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 2; ++out_c_iter) {
                                        t_out[px_idx][out_c_iter] = __hfma2(val ,  t_f[in_c_iter+2][out_c_iter], t_out[px_idx][out_c_iter]);
                                        t_out[px_idx][out_c_iter] = __hfma2(val2 , t_f[in_c_iter+3][out_c_iter], t_out[px_idx][out_c_iter]);
                                    }
                                }

                                {
                                    const half2 val = *reinterpret_cast<const half2*>(&vals4.z);
                                    const half2 val2 = __lowhigh2highlow(val);
                                    #pragma unroll
                                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 2; ++out_c_iter) {
                                        t_out[px_idx][out_c_iter] = __hfma2(val ,  t_f[in_c_iter+4][out_c_iter], t_out[px_idx][out_c_iter]);
                                        t_out[px_idx][out_c_iter] = __hfma2(val2 , t_f[in_c_iter+5][out_c_iter], t_out[px_idx][out_c_iter]);
                                    }
                                }

                                {
                                    const half2 val = *reinterpret_cast<const half2*>(&vals4.w);
                                    const half2 val2 = __lowhigh2highlow(val);
                                    #pragma unroll
                                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 2; ++out_c_iter) {
                                        t_out[px_idx][out_c_iter] = __hfma2(val ,  t_f[in_c_iter+6][out_c_iter], t_out[px_idx][out_c_iter]);
                                        t_out[px_idx][out_c_iter] = __hfma2(val2 , t_f[in_c_iter+7][out_c_iter], t_out[px_idx][out_c_iter]);
                                    }
                                }
                            }
                        }
                    }
                } 
            }
            __syncthreads();
        }

        if (out_c_off + lane_idx*OUT_C_PER_THREAD < dim.out.c) {
            #pragma unroll
            for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                const int gl_px_idx = px_idx * n_warps + warp_idx;
                const int out_y = gl_px_idx / w_in;
                const int out_x = gl_px_idx % w_in;
                const int out_y_im = out_y + tile_start_out_y;
                const int out_x_im = out_x + tile_start_out_x;
                const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                if (valid) {
                    #pragma unroll
                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD/2; out_c_iter += 4) {
                        const int out_c_gl = out_c_iter * 8 * WARP_SIZE + lane_idx * 8 + out_c_off;
                        // for some reason, uncoalesced write seems to be faster here
                        if (out_c_gl + 7 < dim.out.c && dim.out.c % 8 == 0) {
                            float4 val = make_float4(
                                *reinterpret_cast<float*>(&t_out[px_idx][out_c_iter]), 
                                *reinterpret_cast<float*>(&t_out[px_idx][out_c_iter + 1]), 
                                *reinterpret_cast<float*>(&t_out[px_idx][out_c_iter + 2]),
                                *reinterpret_cast<float*>(&t_out[px_idx][out_c_iter + 3])
                            );
                            *reinterpret_cast<float4*>(&batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl]) = val;
                        } 
                        else 
                        if (out_c_gl < dim.out.c) {
                            if (dim.out.c % 2 == 0) {
                                for (int chunk_iter = 0; chunk_iter < 4; chunk_iter++) {
                                    const int out_c_gl2 = out_c_gl + chunk_iter * 2;
                                    if (out_c_gl2 < dim.out.c) {
                                        *reinterpret_cast<half2*>(&batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl2]) = t_out[px_idx][out_c_iter + chunk_iter];
                                    }
                                }
                            } 
                            else {
                                for (int chunk_iter = 0; chunk_iter < 4; chunk_iter++) {
                                    const int out_c_gl2 = out_c_gl + chunk_iter * 2;
                                    if (out_c_gl2 < dim.out.c) {
                                        if (out_c_gl2 + 1 < dim.out.c) {
                                            batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl2 + 1] = __high2half(t_out[px_idx][out_c_iter + chunk_iter]);
                                        }
                                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl2] = __low2half(t_out[px_idx][out_c_iter + chunk_iter]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


template<typename scalar_t = half, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, int OUT_C_PER_THREAD=8, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1>
__global__ void deltacnn_1x1_hp_2(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* filter,
    const scalar_t* bias,
    const uint32_t* mask,
    uint32_t* out_mask,
    Dimensions dim,
    ConvConfig config
) {
    const int KERNEL_SIZE = 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, false>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);

    const uint8_t w_in = pixelsPerBlockX;
    const uint8_t n_in_px = pixelsPerBlockX * pixelsPerBlockY;
    const uint8_t n_in_px_aligned = divup(n_in_px, 4) * 4;
    const int in_row_vals = dim.in.w * dim.in.c;

    const uint8_t lane_idx = threadIdx.x % WARP_SIZE;
    const uint8_t warp_idx = threadIdx.x / WARP_SIZE;
    const uint8_t n_warps = BLOCK_SIZE / WARP_SIZE;

    const uint8_t s_in_channels = 16;
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const uint8_t out_px_per_thread = divup(n_pixels_out, n_warps);

    const uint16_t out_c_aligned = divup(dim.out.c,2)*2;

    __shared__ half2 s_in[n_in_px_aligned][WARP_SIZE];
     uint32_t* s_mask = reinterpret_cast<uint32_t*>(s_in);
    __shared__ half2 s_f[s_in_channels][OUT_C_PER_THREAD * WARP_SIZE];
    uint64_t t_mask = 0LLU;
    uint32_t density;

#ifdef ENABLE_METRICS
    if (tile_start_z == 0 && threadIdx.x == 0) {
        if (DCMetrics::track_filter_reads) {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c) + KERNEL_SIZE*KERNEL_SIZE*dim.out.c*dim.in.c/config.groups);
        } else {
            atomicAdd(&d_metrics->n_vals_read_dense, uint64_t(n_in_px * dim.in.c)); 
        }
        atomicAdd(&d_metrics->n_vals_written_dense, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    load_mask<BLOCK_SIZE, n_in_px, w_in, false, KERNEL_SIZE, STRIDE>(tile_start_in_y, tile_start_in_x, lane_idx, batch_mask, &s_mask[0], t_mask, density, dim, config);

    if (out_mask != nullptr && tile_start_z == 0) {
        write_mask<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, false, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
    }
#ifdef ENABLE_METRICS
    else if (tile_start_z == 0 && threadIdx.x == 0) {
        atomicAdd(&d_metrics->n_vals_written, uint64_t(n_pixels_out * dim.out.c)); 
    }
#endif

    if (density == 0) {
        // nothing to do here. set everything to zero and leave
        if (config.set_sparse_zero) {
            set_out_zero_hp<scalar_t, BLOCK_SIZE, n_pixels_out, pixelsPerBlockX, OUT_CHANNELS_PER_BLOCK, FULL_DEPTH, false>(
                batch_out, bias, tile_start_z, tile_start_out_y, tile_start_out_x, dim, config
            );
        }
        return;
    }

    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += OUT_C_PER_THREAD * WARP_SIZE) {
        half2 t_out[out_px_per_thread][OUT_C_PER_THREAD/2];
        if (dim.out.c % 2 == 0) {
            #pragma unroll
            for (int out_c_idx = 0; out_c_idx < OUT_C_PER_THREAD/2; ++out_c_idx) {
                const int out_c_gl = out_c_off + lane_idx * OUT_C_PER_THREAD + out_c_idx * 2;
                if (out_c_gl + 1 < dim.out.c) {
                    const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : *reinterpret_cast<const half2*>(&bias[out_c_gl]);
                    #pragma unroll
                    for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        t_out[px_idx][out_c_idx] = t_bias;
                    }
                } else {
                    const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : __halves2half2(bias[out_c_gl], __float2half(0.0f));
                    #pragma unroll
                    for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        t_out[px_idx][out_c_idx] = t_bias;
                    }
                }
            }
        } else 
        {
            #pragma unroll
            for (int out_c_idx = 0; out_c_idx < OUT_C_PER_THREAD/2; ++out_c_idx) {
                const int out_c_gl = out_c_off + lane_idx * OUT_C_PER_THREAD + out_c_idx * 2;
                if (out_c_gl + 1 < dim.out.c) {
                    const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : __halves2half2(bias[out_c_gl], bias[out_c_gl + 1]);
                    #pragma unroll
                    for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        t_out[px_idx][out_c_idx] = t_bias;
                    }
                } else {
                    const half2 t_bias = bias == nullptr ? __half2half2(__float2half(0.0f)) : __halves2half2(bias[out_c_gl], __float2half(0.0f));
                    #pragma unroll
                    for (int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        t_out[px_idx][out_c_idx] = t_bias;
                    }
                }
            }
        }

        for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE*2) {
            // load inputs            
            __syncthreads();
            for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                const int in_y = px_idx / w_in; 
                const int in_y_im = in_y * STRIDE + tile_start_in_y;
                const int in_x = px_idx % w_in;
                const int in_x_im = in_x * STRIDE + tile_start_in_x; 
                const int in_c = in_c_off + lane_idx * 2;
                const bool valid = (t_mask & (1LLU << (in_y * w_in + in_x))) && in_c < dim.in.c;
                
                if (!valid) {
                    s_in[in_y * w_in + in_x][lane_idx] = __half2half2(__float2half(0.0f));
                }
                else {
                    if (in_c + 1 < dim.in.c) {
                        if (dim.in.c % 2 == 0) {
                            s_in[in_y * w_in + in_x][lane_idx] =
                                *reinterpret_cast<const half2*>(&batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c]);
                        } else {
                            s_in[in_y * w_in + in_x][lane_idx] = 
                                __halves2half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c + 1]);
                        }
                    } else {
                        s_in[in_y * w_in + in_x][lane_idx] =
                            __halves2half2(batch_in[in_y_im * in_row_vals + in_x_im * dim.in.c + in_c], __float2half(0.0f));
                    }
                }
            }

            const int in_c_per_iter = 8;
            for(int in_c = 0; in_c < WARP_SIZE * 2 && in_c + in_c_off < dim.in.c; in_c += in_c_per_iter) {
                if (in_c % s_in_channels == 0) {
                    __syncthreads();
                    // load filters
                    for (int in_c_idx = warp_idx*2; in_c_idx < s_in_channels; in_c_idx += n_warps*2) {
                        for (int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD/2; ++out_c_iter) {
                            int in_c_global = in_c_idx + in_c_off + in_c;
                            const int out_c_global = out_c_off + lane_idx * OUT_C_PER_THREAD + out_c_iter * 2;
                            const bool valid = in_c_global < dim.in.c && out_c_global < dim.out.c;
                            
                            s_f[in_c_idx][lane_idx * OUT_C_PER_THREAD/2 + out_c_iter] = valid ? 
                                *reinterpret_cast<const half2*>(&filter[in_c_global * out_c_aligned + out_c_global]) : 
                                __half2half2(__float2half(0.0f));
                            s_f[in_c_idx+1][lane_idx * OUT_C_PER_THREAD/2 + out_c_iter] = valid ? 
                                *reinterpret_cast<const half2*>(&filter[(in_c_global+1) * out_c_aligned + out_c_global]) : 
                                __half2half2(__float2half(0.0f));
                        }
                    }
                    __syncthreads();
                }

                
                if (out_c_off + lane_idx*OUT_C_PER_THREAD < dim.out.c) {
                    half2 t_f[in_c_per_iter][OUT_C_PER_THREAD/2];

                    #pragma unroll
                    for (int in_c_iter = 0; in_c_iter < in_c_per_iter; ++in_c_iter) {
                        #pragma unroll
                        for (int t_out_c = 0; t_out_c < OUT_C_PER_THREAD / 2; ++t_out_c) {
                            t_f[in_c_iter][t_out_c] = s_f[(in_c % s_in_channels) + in_c_iter][lane_idx * OUT_C_PER_THREAD / 2 + t_out_c];
                        }
                    }

                    #pragma unroll
                    for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                        #pragma unroll
                        for (int in_c_iter=0; in_c_iter < in_c_per_iter; in_c_iter += 2) {
                            const half2 val = *reinterpret_cast<const half2*>(&s_in[px_idx * n_warps + warp_idx][(in_c + in_c_iter) / 2]);
                            const half2 val2 = __lowhigh2highlow(val);
                            #pragma unroll
                            for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD / 2; ++out_c_iter) {
                                t_out[px_idx][out_c_iter] = __hfma2(val ,  t_f[in_c_iter][out_c_iter], t_out[px_idx][out_c_iter]);
                                t_out[px_idx][out_c_iter] = __hfma2(val2 , t_f[in_c_iter+1][out_c_iter], t_out[px_idx][out_c_iter]);
                            }
                        }
                    }
                } 
            }
            __syncthreads();
        }

        if (out_c_off + lane_idx*OUT_C_PER_THREAD < dim.out.c) {
            #pragma unroll
            for(int px_idx = 0; px_idx < out_px_per_thread; ++px_idx) {
                const int gl_px_idx = px_idx * n_warps + warp_idx;
                const int out_y = gl_px_idx / w_in;
                const int out_x = gl_px_idx % w_in;
                const int out_y_im = out_y + tile_start_out_y;
                const int out_x_im = out_x + tile_start_out_x;
                const bool valid = out_y_im < dim.out.h && out_x_im < dim.out.w;
                if (valid) {
                    #pragma unroll
                    for(int out_c_iter = 0; out_c_iter < OUT_C_PER_THREAD/2; ++out_c_iter) {
                        const int out_c_gl = out_c_off + lane_idx * OUT_C_PER_THREAD + out_c_iter * 2;

                        if (out_c_gl < dim.out.c) {
                            if (dim.out.c % 2 == 0) {
                                *reinterpret_cast<half2*>(&batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl]) = t_out[px_idx][out_c_iter];
                                    
                            } 
                            else {
                                if (out_c_gl + 1 < dim.out.c) {
                                    batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl + 1] = __high2half(t_out[px_idx][out_c_iter]);
                                }
                                batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c_gl] = __low2half(t_out[px_idx][out_c_iter]);
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename scalar_t = float, bool ENABLE_DILATION>
void deltacnn_3x3_standard(scalar_t *input, scalar_t *output, scalar_t *filter, scalar_t *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config)
{
    if (config.stride[0] == 1 && config.stride[1] == 1) {
        const int stride = 1;
        const int pixelsPerBlockX = 6;
        const int pixelsPerBlockY = 6;
        const int out_channels_per_block = 32;
        uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
        uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];

        dim3 gridDim(x, y, dim.batch_size);

        if (blocks < 50 && dim.out.c >= 96 && dim.out.c % 96 == 0) {
            const uint32_t threadsLarge = 96;
            const int out_channels_per_block_tiled = threadsLarge;
            uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block_tiled);
            dim3 gridDim(x,y,z_dim);
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block_tiled, true, false, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (blocks < 20 && dim.out.c > 128) {
            const uint32_t threadsLarge = 128;
            const int out_channels_per_block_tiled = threadsLarge;
            uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block_tiled);
            dim3 gridDim(x,y,z_dim);
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block_tiled, true, false, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (dim.out.c > 256) {
            const uint32_t threadsLarge = 256;
            const int out_channels_per_block_tiled = threadsLarge;
            uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block_tiled);
            dim3 gridDim(x,y,z_dim);
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block_tiled, true, false, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (dim.out.c > 192) {
            const uint32_t threadsLarge = 256;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (dim.out.c > 128) {
            const uint32_t threadsLarge = 192;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else if (dim.out.c > 96) {
            const uint32_t threadsMedium = 128;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsMedium, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsMedium>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else if (dim.out.c > 64) {
            const uint32_t threadsMedium = 96;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsMedium, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsMedium>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else if (dim.out.c > 32) {
            const uint32_t threadsSmall = 64;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsSmall, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsSmall>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else {
            const uint32_t threadsTiny = 32;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsTiny, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsTiny>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
    } else if (config.stride[0] == 2 && config.stride[1] == 2) {
        const int stride = 2;
        const int pixelsPerBlockX = 3;
        const int pixelsPerBlockY = 3;
        const int out_channels_per_block = 32;
        uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
        uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
        dim3 gridDim(x, y, dim.batch_size);
        
        if (blocks < 50 && dim.out.c >= 96 && dim.out.c % 96 == 0) {
            const uint32_t threadsLarge = 96;
            const int out_channels_per_block_tiled = threadsLarge;
            uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block_tiled);
            dim3 gridDim(x,y,z_dim);
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block_tiled, true, false, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (blocks < 20 && dim.out.c > 128) {
            const uint32_t threadsLarge = 128;
            const int out_channels_per_block_tiled = threadsLarge;
            uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block_tiled);
            dim3 gridDim(x,y,z_dim);
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block_tiled, true, false, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (dim.out.c > 256) {
            const uint32_t threadsLarge = 256;
            const int out_channels_per_block_tiled = threadsLarge;
            uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block_tiled);
            dim3 gridDim(x,y,z_dim);
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block_tiled, true, false, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (dim.out.c > 192) {
            const uint32_t threadsLarge = 256;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
        else if (dim.out.c > 128) {
            const uint32_t threadsLarge = 192;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsLarge, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsLarge>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else if (dim.out.c > 96) {
            const uint32_t threadsMedium = 128;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsMedium, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsMedium>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else if (dim.out.c > 64) {
            const uint32_t threadsMedium = 96;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsMedium, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsMedium>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else if (dim.out.c > 32) {
            const uint32_t threadsSmall = 64;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsSmall, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsSmall>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        } else {
            const uint32_t threadsTiny = 32;
            deltacnn_3x3_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threadsTiny, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threadsTiny>>>(
                input, output, filter, bias, mask, out_mask, dim, config);
        }
    } else {
        printf("Stride other than 1x1 and 2x2 not supported, got %ix%i\n", config.stride[0], config.stride[1]);
        throw "Stride other than 1x1 and 2x2 not supported";
    }
}

template<typename scalar_t = float, bool ENABLE_DILATION>
void deltacnn_dilation(scalar_t *input, scalar_t *output, scalar_t *filter, scalar_t *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    if (config.kernel_size[0] == 3 && config.kernel_size[1] == 3) {
        const int kernel_size = 3;
        if (config.groups == 1) {
            deltacnn_3x3_standard<scalar_t, ENABLE_DILATION>(input, output, filter, bias, mask, out_mask, dim, config);
        } else if (config.groups == dim.out.c && config.groups == dim.in.c) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 4;
                const int pixelsPerBlockY = 4;
                const uint32_t threads = 128;
                const int out_channels_per_block = threads;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);

                if (dim.out.c <= 32) {
                    const uint32_t threads = 32;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 64) {
                    const uint32_t threads = 64;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (blocks * z_dim < 50) {
                    const uint32_t threads = 64;
                    const int out_channels_per_block = threads;
                    uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, z_dim);

                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (z_dim > 1) {
                    uint32_t z_dim = dim.batch_size * divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, z_dim);

                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 3;
                const int pixelsPerBlockY = 3;
                const uint32_t threads = 128;
                const int out_channels_per_block = threads;
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);

                if (dim.out.c <= 32) {
                    const uint32_t threads = 32;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 64) {
                    const uint32_t threads = 32;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 128) {
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } else {
                printf("Stride other than 1x1 and 2x2 not supported, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Stride other than 1x1 and 2x2 not supported";
            }
        } else {
                printf("Groups other than 1 and C_out not supported, got %i\n", config.groups);
                throw "Groups other than 1 and C_out not supported";
        }
    } else if (config.kernel_size[0] == 1 && config.kernel_size[1] == 1) {
        if (config.stride[0] == 1 && config.stride[1] == 1) {
            const int stride = 1;
            const int pixelsPerBlockX = 8;
            const int pixelsPerBlockY = 4;
            const int out_channels_per_block = WARP_SIZE;
            // uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY) * divup(dim.out.w, pixelsPerBlockX);
            
            uint32_t x = divup(dim.out.w, pixelsPerBlockX);
            uint32_t y = divup(dim.out.h, pixelsPerBlockY);
            dim3 gridDim(x, y, dim.batch_size);
            
            const uint32_t threads = 128;
            const bool sub_tile_sparsity = false;
            
            if (dim.out.c <= 32) {
                const int out_channels_per_thread = 1;
                deltacnn_1x1_sp_2<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, true, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            } else if (dim.out.c <= 64) {
                const int out_channels_per_thread = 2;
                deltacnn_1x1_sp_2<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, true, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            } else if (dim.out.c <= 96) {
                const int out_channels_per_thread = 3;
                deltacnn_1x1_sp_2<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, true, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            } else if (dim.out.c <= 128) {
                const int out_channels_per_thread = 4;
                deltacnn_1x1_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, true, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            } else if (dim.out.c <= 192) {
                const int out_channels_per_thread = 3;
                const int out_channels_per_block = WARP_SIZE * out_channels_per_thread;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                dim3 gridDim(x, y, dim.batch_size * z_dim);
                deltacnn_1x1_sp_2<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, false, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            }
            else {
                const int out_channels_per_thread = 4;
                const int out_channels_per_block = WARP_SIZE * out_channels_per_thread;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                dim3 gridDim(x, y, dim.batch_size * z_dim);
                deltacnn_1x1_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, false, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            }
        }
        else if (config.stride[0] == 2 && config.stride[1] == 2) {
            const int stride = 2;
            const int pixelsPerBlockX = 8;
            const int pixelsPerBlockY = 4;
            const int out_channels_per_block = 32;
            
            uint32_t x = divup(dim.out.w, pixelsPerBlockX);
            uint32_t y = divup(dim.out.h, pixelsPerBlockY);
            dim3 gridDim(x, y, dim.batch_size);
            
            const uint32_t threads = 128;
            const bool sub_tile_sparsity = false;
            
            if (dim.out.c > 128) {
                const int out_channels_per_thread = 4;
                const int out_channels_per_block = WARP_SIZE * out_channels_per_thread;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                dim3 gridDim(x, y, dim.batch_size * z_dim);
                deltacnn_1x1_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, false, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            }
            else {
                const int out_channels_per_thread = 4;
                deltacnn_1x1_sp<scalar_t, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_channels_per_thread, sub_tile_sparsity, true, stride><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
            }
        } else {
            printf("Stride other than 1x1 and 2x2 not supported for 1x1 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
            throw "Strides other than 1x1 and 2x2 not supported for 1x1 convolution";
        }
    }
    else if (config.kernel_size[0] == 5 && config.kernel_size[1] == 5) {
        const int kernel_size = 5;
        
        if (config.groups == 1) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 5;
                const int pixelsPerBlockY = 5;

                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);
        
                if (dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128)) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x,y,z_dim * dim.batch_size);
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 256;
                    const int out_channels_per_block = threads;
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 64;
                    const int out_channels_per_block = threads;
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }

            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 5;
                const int pixelsPerBlockY = 5;
                
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);

                if ((dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128))) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x,y,z_dim*dim.batch_size);
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 256;
                    const int out_channels_per_block = threads;
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 64;
                    const int out_channels_per_block = threads;
                    deltacnn_standard_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } else {
                printf("Stride other than 1x1 and 2x2 not supported for 5x5 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1 and 2x2 not supported for 5x5 convolution";
            }
        }
        else if (config.groups == dim.in.c && config.groups == dim.out.c) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 3;
                const int pixelsPerBlockY = 3;

                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);
        
                if (dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128)) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x,y,z_dim * dim.batch_size);
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 256;
                    const int out_channels_per_block = threads;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 64;
                    const int out_channels_per_block = threads;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }

            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 2;
                const int pixelsPerBlockY = 2;
                
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0]*config.dilation[1];
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);

                if ((dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128))) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x,y,z_dim*dim.batch_size);
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 256;
                    const int out_channels_per_block = threads;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 128;
                    const int out_channels_per_block = threads;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 64;
                    const int out_channels_per_block = threads;
                    deltacnn_dw_conv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } else {
                printf("Stride other than 1x1 and 2x2 not supported for 5x5 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1 and 2x2 not supported for 5x5 convolution";
            }
        } else {
            printf("Groups must either be 1 or same as channels for 5x5 convolution\n");
            throw "Groups must either be 1 or same as channels for 5x5 convolution";
        }
    }
    else if (config.kernel_size[0] == 7 && config.kernel_size[1] == 7) {
        if (config.groups != 1) {
            printf("7x7 convolution does not support depth wise filters\n");
            throw "7x7 convolution does not support depth wise filters";
        }
        if (config.stride[0] != 2 || config.stride[1] != 2) {
            printf("Stride other than 2x2 not supported for 7x7 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
            throw "Strides other than 2x2 not supported for 7x7 convolution";
        }
        const int stride = 2;
        const int pixelsPerBlockX = 3;
        const int pixelsPerBlockY = 3;
        
        uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
        dim3 gridDim(x, y, dim.batch_size);
        const int threads = 64;
        const int out_channels_per_block = threads;
        deltacnn_standard_conv_sp<scalar_t, 7, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
            input, output, filter, bias, mask, out_mask, dim, config);
    }
    else {
        printf("Kernel sizes other than 7x7, 5x5, 3x3 and 1x1 not supported. got %ix%i\n", config.kernel_size[0], config.kernel_size[1]);
        throw "Kernel sizes other than 7x7, 5x5, 3x3 and 1x1 not supported";
    }
}

template<typename scalar_t = float>
void deltacnn(scalar_t *input, scalar_t *output, scalar_t *filter, scalar_t *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {

    if (config.dilation[0] == 1 && config.dilation[1] == 1) {
        deltacnn_dilation<scalar_t, false>(input, output, filter, bias, mask, out_mask, dim, config);
    } else {
        deltacnn_dilation<scalar_t, true>(input, output, filter, bias, mask, out_mask, dim, config);        
    }
}

template<bool SUB_TILE_SPARSITY, bool ENABLE_DILATION=false>
void deltacnn_hp_templates(half *input, half *output, half *filter, half* bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    const int out_channels_per_block = 32;
    const int out_c_per_thread = 2;

    if (config.kernel_size[0] == 3 && config.kernel_size[1] == 3) {
        const int kernel_size = 3;

        if (config.groups == 1) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 5;
                const int pixelsPerBlockY = 5;
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0] * config.dilation[1];
                
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);

                if (dim.out.c >= 128 && blocks < 50) {
                    const uint32_t threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    const int z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c <= 64) {
                    const uint32_t threads = 32;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 128) {
                    const uint32_t threads = 64;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else
                {
                    const uint32_t threads = 128;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                }
            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 3;
                const int pixelsPerBlockY = 3;
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0] * config.dilation[1];
                
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);


                if (dim.out.c >= 128 && blocks < 50) {
                    const uint32_t threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    const int z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c <= 64) {
                    const uint32_t threads = 64;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 128) {
                    const uint32_t threads = 64;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else 
                {
                    const uint32_t threads = 128;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    // deltacnn_3x3_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                    //     (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } 
            } else {
                    printf("Stride other than 1x1 and 2x2 not supported, got %ix%i\n", config.stride[0], config.stride[1]);
                    throw "Stride other than 1x1 and 2x2 not supported";
            }
        } else if (config.groups == dim.in.c && config.groups == dim.out.c) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 4;
                const int pixelsPerBlockY = 4;
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);
                
                if (dim.out.c <= 64) {
                    const uint32_t threads = 32;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 128) {
                    const uint32_t threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const uint32_t threads = 128;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 3;
                const int pixelsPerBlockY = 3;
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);
                
                if (dim.out.c <= 64) {
                    const uint32_t threads = 32;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c <= 128) {
                    const uint32_t threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const uint32_t threads = 128;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, SUB_TILE_SPARSITY, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } else {
                printf("Stride other than 1x1 and 2x2 not supported, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Stride other than 1x1 and 2x2 not supported";
            }            
        } else {
            printf("Groups only supported for group=1 and group = c_in = c_out, got %i\n", config.groups);
            throw "Groups only supported for group=1 and group = c_in = c_out";
        }
    } else if (config.kernel_size[0] == 1 && config.kernel_size[1] == 1) {
        if (config.stride[0] == 1 && config.stride[1] == 1) {
            const int stride = 1;

            if (dim.out.w <= 32 && dim.out.h <= 32) {
                const int pixelsPerBlockX = 4;
                const int pixelsPerBlockY = 4;
                uint32_t x = divup(dim.out.w, pixelsPerBlockX);
                uint32_t y = divup(dim.out.h, pixelsPerBlockY);

                if (dim.out.c <= 64) {
                    const uint32_t threads = 64;
                    const int out_c_per_thread = 2;
                    const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);

                    if (z_dim > 1) {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    } else {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    }
                } else if (dim.out.c <= 128) {
                    const uint32_t threads = 64;
                    const int out_c_per_thread = 4;
                    const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);

                    if (z_dim > 1) {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    } else {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    }

                } else {
                    const uint32_t threads = 128;
                    const int out_c_per_thread = 8;
                    const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);

                    if (z_dim > 1) {
                        deltacnn_1x1_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    } else {
                        deltacnn_1x1_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    }
                }
            } else {
                const int pixelsPerBlockX = 8;
                const int pixelsPerBlockY = 6;
                uint32_t x = divup(dim.out.w, pixelsPerBlockX);
                uint32_t y = divup(dim.out.h, pixelsPerBlockY);
                if (dim.out.c <= 64) {
                    const uint32_t threads = 128;
                    const int out_c_per_thread = 2;
                    const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);

                    if (z_dim > 1) {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    } else {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    }
                } else if (dim.out.c <= 128) {
                    const uint32_t threads = 128;
                    const int out_c_per_thread = 4;
                    const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);

                    if (z_dim > 1) {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    } else {
                        deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    }

                } else {
                    const uint32_t threads = 256;
                    const int out_c_per_thread = 8;
                    const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, dim.batch_size * z_dim);

                    if (z_dim > 1) {
                        deltacnn_1x1_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    } else {
                        deltacnn_1x1_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                            (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                    }
                }
            }
        } else if (config.stride[0] == 2 && config.stride[1] == 2) {
            const int stride = 2;
            const int pixelsPerBlockX = 8;
            const int pixelsPerBlockY = 6;
            uint32_t x = divup(dim.out.w, pixelsPerBlockX);
            uint32_t y = divup(dim.out.h, pixelsPerBlockY);
            
            if (dim.out.c <= 64) {
                const uint32_t threads = 64;
                const int out_c_per_thread = 2;
                const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                dim3 gridDim(x, y, dim.batch_size * z_dim);

                if (z_dim > 1) {
                    deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else {
                    deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                }
            } else if (dim.out.c <= 128) {
                const uint32_t threads = 64;
                const int out_c_per_thread = 4;
                const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                dim3 gridDim(x, y, dim.batch_size * z_dim);

                if (z_dim > 1) {
                    deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else {
                    deltacnn_1x1_hp_2<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                }

            } else {
                const uint32_t threads = 128;
                const int out_c_per_thread = 8;
                const int out_channels_per_block = WARP_SIZE * out_c_per_thread;
                uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                dim3 gridDim(x, y, dim.batch_size * z_dim);

                if (z_dim > 1) {
                    deltacnn_1x1_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, false, stride><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                } else {
                    deltacnn_1x1_hp<half, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, out_c_per_thread, SUB_TILE_SPARSITY, true, stride><<<gridDim, threads>>>(
                        (half*)input, (half*)output, (half*)filter, (half*) bias, mask, out_mask, dim, config);
                }
            }
        } else {
            printf("Stride other than 1x1 and 2x2 not supported for 1x1 conv, got %ix%i\n", config.stride[0], config.stride[1]);
            throw "Stride other than 1x1 and 2x2 not supported for 1x1 conv";
        }
    } else if (config.kernel_size[0] == 5 && config.kernel_size[1] == 5) {
        const int kernel_size = 5;
        if (config.groups == 1) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 5;
                const int pixelsPerBlockY = 5;
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0] * config.dilation[1];
                
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);
        
                if (dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128)) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x,y,z_dim*dim.batch_size);
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 128;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 32;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } 
            else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 5;
                const int pixelsPerBlockY = 5;
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY) * divup(dim.out.w, pixelsPerBlockX);
                
                uint32_t x = divup(dim.out.w, pixelsPerBlockX);
                uint32_t y = divup(dim.out.h, pixelsPerBlockY);
                dim3 gridDim(x, y, dim.batch_size);
        
                if ((dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128))) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, z_dim * dim.batch_size);
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 128;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 32;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_standard_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } 
            else {
                printf("Stride other than 1x1 and 2x2 not supported for 5x5 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1 and 2x2 not supported for 5x5 convolution";
            }
        }
        else if (config.groups == dim.in.c && config.groups == dim.out.c) {
            if (config.stride[0] == 1 && config.stride[1] == 1) {
                const int stride = 1;
                const int pixelsPerBlockX = 3;
                const int pixelsPerBlockY = 3;
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY*config.dilation[0]) * divup(dim.out.w, pixelsPerBlockX*config.dilation[1]) * config.dilation[0] * config.dilation[1];
                
                uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
                uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
                dim3 gridDim(x, y, dim.batch_size);
        
                if (dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128)) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x,y,z_dim*dim.batch_size);
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 128;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 32;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } 
            else if (config.stride[0] == 2 && config.stride[1] == 2) {
                const int stride = 2;
                const int pixelsPerBlockX = 2;
                const int pixelsPerBlockY = 2;
                uint32_t blocks = dim.batch_size * divup(dim.out.h, pixelsPerBlockY) * divup(dim.out.w, pixelsPerBlockX);
                
                uint32_t x = divup(dim.out.w, pixelsPerBlockX);
                uint32_t y = divup(dim.out.h, pixelsPerBlockY);
                dim3 gridDim(x, y, dim.batch_size);
        
                if ((dim.out.c > 256 || (blocks < 50 && dim.out.c >= 128))) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    uint32_t z_dim = divup(dim.out.c, out_channels_per_block);
                    dim3 gridDim(x, y, z_dim * dim.batch_size);
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
                else if (dim.out.c > 128) {
                    const int threads = 128;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else if (dim.out.c > 64) {
                    const int threads = 64;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                } else {
                    const int threads = 32;
                    const int out_channels_per_block = threads * out_c_per_thread;
                    deltacnn_dw_conv_hp<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
                        input, output, filter, bias, mask, out_mask, dim, config);
                }
            } 
            else {
                printf("Stride other than 1x1 and 2x2 not supported for 5x5 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
                throw "Strides other than 1x1 and 2x2 not supported for 5x5 convolution";
            }
        } else {
            printf("Groups must either be 1 or same as channels for 5x5 convolution\n");
            throw "Groups must either be 1 or same as channels for 5x5 convolution";
        }
    }
    else if (config.kernel_size[0] == 7 && config.kernel_size[1] == 7) {
        if (config.groups != 1) {
            printf("7x7 convolution does not support depth wise filters\n");
            throw "7x7 convolution does not support depth wise filters";
        }
        if (config.stride[0] != 2 || config.stride[1] != 2) {
            printf("Stride other than 2x2 not supported for 7x7 convolution, got %ix%i\n", config.stride[0], config.stride[1]);
            throw "Strides other than 2x2 not supported for 7x7 convolution";
        }
        const int stride = 2;
        const int pixelsPerBlockX = 3;
        const int pixelsPerBlockY = 3;
        
        uint32_t x = divup(dim.out.w, pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        uint32_t y = divup(dim.out.h, pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
        dim3 gridDim(x, y, dim.batch_size);
        const int threads = 64;
        const int out_channels_per_block = threads * 2;
        deltacnn_standard_conv_hp<half, 7, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
            input, output, filter, bias, mask, out_mask, dim, config);
    } 
    else {
        printf("Kernel sizes other than 7x7, 5x5, 3x3 and 1x1 not supported, got %ix%i\n", config.kernel_size[0], config.kernel_size[1]);
        throw "Kernel sizes other than 7x7, 5x5, 3x3 and 1x1 not supported";
    }
}

void deltacnn_hp(half *input, half *output, half *filter, half* bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    if (config.dilation[0] == 1 && config.dilation[1] == 1) {
        deltacnn_hp_templates<true, false>(input, output, filter, bias, mask, out_mask, dim, config);
    } else {
        deltacnn_hp_templates<true, true>(input, output, filter, bias, mask, out_mask, dim, config);
    }
    if (!config.sub_tile_sparsity) {
        printf("INFO: subtile sparsity is always enabled\n");
    }
}

template void deltacnn<float>(float *input, float *output, float *filter, float *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config);