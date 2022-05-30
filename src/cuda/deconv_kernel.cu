// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "deconv_kernel.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Utility.cuh"

#define divup(a, b) (((a) + (b) - 1) / (b))

__device__ DCMetrics* d_metrics;

void init_d_metrics_deconv_kernels() {
#ifdef ENABLE_METRICS
    copy_performance_metrics_to_gpu(d_metrics);
#endif
}


template<int pixelsPerBlockX, int pixelsPerBlockY, int OUT_CHANNELS_PER_BLOCK, int STRIDE, bool FULL_DEPTH, bool ENABLE_DILATION, int KERNEL_SIZE>
__device__ __forceinline__ void calc_tile_indices_deconv(int& tile_start_out_y, int& tile_start_out_x, int& tile_start_in_y, int& tile_start_in_x, int& tile_start_z, int& batch, const ConvConfig& config, const Dimensions& dim) {
    if (ENABLE_DILATION) {
        printf("ERROR: Transposed convolution does not support dilation.\n");
        tile_start_out_y = (blockIdx.y / config.dilation[0]) * pixelsPerBlockY * config.dilation[0] + (blockIdx.y % config.dilation[0]);
        tile_start_out_x = (blockIdx.x / config.dilation[1]) * pixelsPerBlockX * config.dilation[1] + (blockIdx.x % config.dilation[1]);
        tile_start_in_y = tile_start_out_y / STRIDE;
        tile_start_in_x = tile_start_out_x / STRIDE;

    } else {
        tile_start_out_y = blockIdx.y * pixelsPerBlockY - config.padding[0];
        tile_start_out_x = blockIdx.x * pixelsPerBlockX - config.padding[1];
        tile_start_in_y = blockIdx.y * pixelsPerBlockY / STRIDE -(divup(KERNEL_SIZE, STRIDE) - 1);
        tile_start_in_x = blockIdx.x * pixelsPerBlockX / STRIDE -(divup(KERNEL_SIZE, STRIDE) - 1);
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
    for (int i = threadIdx.x; i < n_in_px; i += BLOCK_SIZE) {
        int y = tile_start_in_y + (i / w_in);
        int x = tile_start_in_x + (i % w_in);
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



template<int w_in, int n_in_px>
__device__ __forceinline__ bool is_px_mask_set(const int x, const int y, const uint64_t t_mask, const uint32_t *s_mask) {
    return n_in_px <= 64 ? ((t_mask & (1LLU << (y * w_in + x))) != 0) : (s_mask[y * w_in + x] != 0);
}


template<int BLOCK_SIZE, int pixelsPerBlockX, int n_pixels_out, int w_in, int n_in_px, int STRIDE, bool ENABLE_DILATION, int KERNEL_SIZE=3> 
__device__ __forceinline__ void write_mask_deconv(uint32_t* out_mask, const int batch, const uint64_t t_mask, const uint32_t *s_mask, const int density, const int tile_start_out_y, const int tile_start_out_x, const Dimensions &dim, const ConvConfig &config) {
    const int DILATION_X = ENABLE_DILATION ? config.dilation[1] : 1;
    const int DILATION_Y = ENABLE_DILATION ? config.dilation[0] : 1;
    uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];

    for (int out_px = threadIdx.x; out_px < n_pixels_out; out_px += BLOCK_SIZE) {
        int p_y = out_px / pixelsPerBlockX;
        int p_x = out_px % pixelsPerBlockX;
        int out_y = p_y * DILATION_Y + tile_start_out_y;
        int out_x = p_x * DILATION_X + tile_start_out_x; 
        if (out_y < 0 || out_y >= dim.out.h || out_x < 0 || out_x >= dim.out.w)
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
                for (int y = 0; y <= KERNEL_SIZE; y++) {
                    for (int x = 0; x <= KERNEL_SIZE; x++) {
                        const int mask_x = (p_x - x / STRIDE);
                        const int mask_y = (p_y - y / STRIDE);
                        const int mask_idx = mask_y * w_in + mask_x;
                        const bool valid = is_px_mask_set<w_in, n_in_px>(mask_x, mask_y, t_mask, s_mask);
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
                atomicAdd(&d_metrics->n_active_flops, uint64_t(KERNEL_SIZE * KERNEL_SIZE / STRIDE / STRIDE * dim.in.c * dim.out.c / config.groups)); 
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
        atomicAdd(&d_metrics->n_dense_flops, uint64_t(n_pixels_out * (KERNEL_SIZE * KERNEL_SIZE / STRIDE / STRIDE * dim.in.c * dim.out.c / config.groups))); 
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


template<typename scalar_t=float, int KERNEL_SIZE=4, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void delta_deconv_sp(
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

    const int IN_PIXELS_OFFSET = divup(KERNEL_SIZE, STRIDE) - 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices_deconv<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION, KERNEL_SIZE>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int w_in = pixelsPerBlockX / STRIDE + IN_PIXELS_OFFSET;
    const int h_in = pixelsPerBlockY / STRIDE + IN_PIXELS_OFFSET;
    const int n_in_px = w_in * h_in;
    const int in_row_vals = dim.in.w * dim.in.c;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    // const int sub_warp_idx = lane_idx / 8;
    // const int sub_warp_lane_idx = lane_idx % 8;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ uint32_t s_mask[n_in_px];
    __shared__ scalar_t s_in[n_in_px][WARP_SIZE];
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
        // TODO implement write mask for deconv!
        // write_mask_deconv<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
        
        if (threadIdx.x == 0) {
            int t_out[n_pixels_out];
            uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];


#ifdef ENABLE_METRICS
            atomicAdd(&d_metrics->n_dense_flops, uint64_t(KERNEL_SIZE*KERNEL_SIZE/STRIDE/STRIDE * n_pixels_out * dim.in.c * dim.out.c / config.groups)); 
            int n_active_inputs = 0;
#endif

            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                t_out[i] = 0;
            }
            const int IN_OFF = 0;
            #pragma unroll
            for (int in_y = IN_OFF; in_y < h_in + IN_OFF; ++in_y) {
                #pragma unroll
                for (int in_x = IN_OFF; in_x < w_in + IN_OFF; ++in_x) {
                    const int val = is_px_mask_set<w_in, n_in_px>(in_x-IN_OFF, in_y-IN_OFF, t_mask, s_mask);

                    const int min_f_y = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_y)*STRIDE);
                    const int min_f_x = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_x)*STRIDE);
                    const int max_f_y = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockY - (in_y-IN_PIXELS_OFFSET)*STRIDE);
                    const int max_f_x = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockX - (in_x-IN_PIXELS_OFFSET)*STRIDE);
                    #pragma unroll
                    for (int f_y = min_f_y; f_y < max_f_y; ++f_y) {
                        #pragma unroll
                        for (int f_x = min_f_x; f_x < max_f_x; ++f_x) {
                            const int out_y = f_y + (in_y-IN_PIXELS_OFFSET) * STRIDE;
                            const int out_x = f_x + (in_x-IN_PIXELS_OFFSET) * STRIDE;
                            const int t_out_idx = out_y * pixelsPerBlockX + out_x;
                            t_out[t_out_idx] += val;
#ifdef ENABLE_METRICS
                            n_active_inputs += val;
#endif
                        }
                    }
                }
            }

#ifdef ENABLE_METRICS
            int n_px_active = 0;
#endif
            #pragma unroll
            for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                #pragma unroll
                for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                    const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                    const bool valid = out_y_im >= 0 && out_y_im < dim.out.h && out_x_im >= 0 && out_x_im < dim.out.w;
                    if (valid) {
#ifdef ENABLE_METRICS
                        ++n_px_active;
#endif
                        batch_out_mask[out_y_im * dim.out.w + out_x_im] = t_out[out_y*pixelsPerBlockX + out_x] > 0 ? 1 : 0;
                    }
                }
            }

#ifdef ENABLE_METRICS
            if (n_px_active > 0) {
                atomicAdd(&d_metrics->n_active_flops, uint64_t(KERNEL_SIZE*KERNEL_SIZE/STRIDE/STRIDE * n_pixels_out * dim.in.c * dim.out.c / config.groups)); 
                atomicAdd(&d_metrics->n_theoretical_flops, uint64_t(n_active_inputs * dim.in.c * dim.out.c / config.groups)); 
            }
#endif
        }
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
            // TODO check why this is not working yet? cuda yells that memory is misaligned
            // if (dim.in.c % 4 == 0) {
            //     for (int px_idx = warp_idx * 4 + sub_warp_idx; px_idx < n_in_px; px_idx += n_warps*4) {
            //         const int in_y = px_idx / w_in; 
            //         const int in_x = px_idx % w_in;
            //         const int in_c = in_c_off + sub_warp_lane_idx * 4;
            //         // const bool valid = (t_mask & (1LLU << (in_y * w_in + in_x))) && in_c < dim.in.c;
            //         const bool valid = 
            //             in_y + tile_start_in_y >= 0 && in_x + tile_start_in_x >= 0 &&
            //             in_y + tile_start_in_y < dim.in.h && in_x + tile_start_in_x < dim.in.w &&
            //             in_c < dim.in.c && 
            //             is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);
                    
            //         if (valid) {
            //             const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
            //             const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
            //             const float4 val = reinterpret_cast<const float4*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 4];
            //             reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = val;
            //         } else {
            //             s_in[in_y * w_in + in_x][sub_warp_lane_idx*4] = 0.0f;
            //             s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 1] = 0.0f;
            //             s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 2] = 0.0f;
            //             s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 3] = 0.0f;
            //         }
            //     }
            // } else 
            if (dim.in.c == 3) {
                for (int val_idx = threadIdx.x; val_idx < n_in_px * 3; val_idx += BLOCK_SIZE) {
                    const int px_idx = val_idx / 3;
                    const int in_c = val_idx % 3;
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const bool valid = 
                        in_y + tile_start_in_y >= 0 && in_x + tile_start_in_x >= 0 &&
                        in_y + tile_start_in_y < dim.in.h && in_x + tile_start_in_x < dim.in.w &&
                        is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

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
                    const bool valid = 
                        in_y + tile_start_in_y >= 0 && in_x + tile_start_in_x >= 0 &&
                        in_y + tile_start_in_y < dim.in.h && in_x + tile_start_in_x < dim.in.w &&
                        in_c < dim.in.c && 
                        is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

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
                            t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[(f_y * KERNEL_SIZE + f_x) * dim.out.c];
                        }
                    }
                
                    const int IN_OFF = 0;
                    #pragma unroll
                    for (int in_y = IN_OFF; in_y < h_in + IN_OFF; ++in_y) {
                        #pragma unroll
                        for (int in_x = IN_OFF; in_x < w_in + IN_OFF; ++in_x) {
                            const scalar_t val = s_in[(in_y-IN_OFF) * w_in + (in_x-IN_OFF)][in_c];
                            const int min_f_y = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_y)*STRIDE);
                            const int min_f_x = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_x)*STRIDE);
                            const int max_f_y = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockY - (in_y-IN_PIXELS_OFFSET)*STRIDE);
                            const int max_f_x = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockX - (in_x-IN_PIXELS_OFFSET)*STRIDE);
                            #pragma unroll
                            for (int f_y = min_f_y; f_y < max_f_y; ++f_y) {
                                #pragma unroll
                                for (int f_x = min_f_x; f_x < max_f_x; ++f_x) {
                                    const int out_y = f_y + (in_y-IN_PIXELS_OFFSET) * STRIDE;
                                    const int out_x = f_x + (in_x-IN_PIXELS_OFFSET) * STRIDE;
                                    const int t_out_idx = out_y * pixelsPerBlockX + out_x;
                                    t_out[t_out_idx] += val * t_f[f_y*KERNEL_SIZE + f_x];
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
                    const bool valid = out_y_im >= 0 && out_y_im < dim.out.h && out_x_im >= 0 && out_x_im < dim.out.w;
                    if (valid) {
                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = t_out[out_y*pixelsPerBlockX + out_x];
                    }
                }
            }
        }
    }
}





template<typename scalar_t=half, int KERNEL_SIZE=4, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int BLOCK_SIZE=256, int OUT_CHANNELS_PER_BLOCK=32, bool SUB_TILE_SPARSITY=false, bool FULL_DEPTH=false, int STRIDE=1, bool ENABLE_DILATION=false>
__global__ void delta_deconv_hp_kernel(
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

    const int IN_PIXELS_OFFSET = divup(KERNEL_SIZE, STRIDE) - 1;

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices_deconv<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, FULL_DEPTH, ENABLE_DILATION, KERNEL_SIZE>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, config, dim);

    scalar_t* batch_out = output + (batch * dim.out.h * dim.out.w * dim.out.c);
    const scalar_t* batch_in = input + (batch * dim.in.h * dim.in.w * dim.in.c);
    const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * dim.in.h * dim.in.w);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int K_HALF = (KERNEL_SIZE-1) / 2;
    const int w_in = pixelsPerBlockX / STRIDE + IN_PIXELS_OFFSET;
    const int h_in = pixelsPerBlockY / STRIDE + IN_PIXELS_OFFSET;
    const int n_in_px = w_in * h_in;
    const int in_row_vals = dim.in.w * dim.in.c;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    const uint16_t out_c_aligned = divup(dim.out.c, 2) * 2;

    __shared__ uint32_t s_mask[n_in_px];
    __shared__ half2 s_in[n_in_px][WARP_SIZE];
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
        // TODO implement write mask for deconv!
        // write_mask_deconv<BLOCK_SIZE, pixelsPerBlockX, n_pixels_out, w_in, n_in_px, STRIDE, ENABLE_DILATION, KERNEL_SIZE>(out_mask, batch, t_mask, &s_mask[0], density, tile_start_out_y, tile_start_out_x, dim, config);
        
        if (threadIdx.x == 0) {
            int t_out[n_pixels_out];
            uint32_t *batch_out_mask = &out_mask[batch * dim.out.w * dim.out.h];


#ifdef ENABLE_METRICS
            atomicAdd(&d_metrics->n_dense_flops, uint64_t(KERNEL_SIZE*KERNEL_SIZE/STRIDE/STRIDE * n_pixels_out * dim.in.c * dim.out.c / config.groups)); 
            int n_active_inputs = 0;
#endif

            #pragma unroll
            for (int i = 0; i < n_pixels_out; ++i) {
                t_out[i] = 0;
            }
            // const int IN_OFF = -(KERNEL_SIZE/STRIDE) + 1;
            const int IN_OFF = 0;
            #pragma unroll
            for (int in_y = IN_OFF; in_y < h_in + IN_OFF; ++in_y) {
                #pragma unroll
                for (int in_x = IN_OFF; in_x < w_in + IN_OFF; ++in_x) {
                    const int val = is_px_mask_set<w_in, n_in_px>(in_x-IN_OFF, in_y-IN_OFF, t_mask, s_mask);
                    const int min_f_y = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_y)*STRIDE);
                    const int min_f_x = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_x)*STRIDE);
                    const int max_f_y = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockY - (in_y-IN_PIXELS_OFFSET)*STRIDE);
                    const int max_f_x = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockX - (in_x-IN_PIXELS_OFFSET)*STRIDE);
                    #pragma unroll
                    for (int f_y = min_f_y; f_y < max_f_y; ++f_y) {
                        #pragma unroll
                        for (int f_x = min_f_x; f_x < max_f_x; ++f_x) {
                            const int out_y = f_y + (in_y-IN_PIXELS_OFFSET) * STRIDE;
                            const int out_x = f_x + (in_x-IN_PIXELS_OFFSET) * STRIDE;
                            const int t_out_idx = out_y * pixelsPerBlockX + out_x;
                            t_out[t_out_idx] += val;
#ifdef ENABLE_METRICS
                            n_active_inputs += val;
#endif
                        }
                    }
                }
            }
        

#ifdef ENABLE_METRICS
            int n_px_active = 0;
#endif
            #pragma unroll
            for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                const int out_y_im = out_y * DILATION_Y + tile_start_out_y;
                #pragma unroll
                for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                    const int out_x_im = out_x * DILATION_X + tile_start_out_x;
                    const bool valid = out_y_im >= 0 && out_y_im < dim.out.h && out_x_im >= 0 && out_x_im < dim.out.w;
                    if (valid) {
#ifdef ENABLE_METRICS
                        ++n_px_active;
#endif
                        batch_out_mask[out_y_im * dim.out.w + out_x_im] = t_out[out_y*pixelsPerBlockX + out_x] > 0 ? 1 : 0;
                    }
                }
            }

#ifdef ENABLE_METRICS
            if (n_px_active > 0) {
                atomicAdd(&d_metrics->n_active_flops, uint64_t(KERNEL_SIZE*KERNEL_SIZE/STRIDE/STRIDE * n_pixels_out * dim.in.c * dim.out.c / config.groups)); 
                atomicAdd(&d_metrics->n_theoretical_flops, uint64_t(n_active_inputs * dim.in.c * dim.out.c / config.groups)); 
            }
#endif
        }
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

    for (int out_c_off = tile_start_z; out_c_off < dim.out.c && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE * 2) {
        const int out_c = out_c_off + threadIdx.x * 2;
        half2 t_out[n_pixels_out];
        const half2 t_bias = bias == nullptr || out_c >= dim.out.c ? __half2half2(__float2half(0.0f)) : *reinterpret_cast<const half2*>(&bias[out_c]);

        #pragma unroll
        for (int i = 0; i < n_pixels_out; ++i) {
            t_out[i] = t_bias;
        }

        for (int in_c_off = 0; in_c_off < dim.in.c; in_c_off += WARP_SIZE * 2) {
            __syncthreads();
            
            // TODO check why this does not work
            // if (dim.in.c % 2 == 0) {
            //     for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
            //         const int in_y = px_idx / w_in; 
            //         const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
            //         const int in_x = px_idx % w_in;
            //         const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
            //         const int in_c = in_c_off + lane_idx * 2;
            //         const bool valid = 
            //             in_y + tile_start_in_y >= 0 && in_x + tile_start_in_x >= 0 &&
            //             in_y + tile_start_in_y < dim.in.h && in_x + tile_start_in_x < dim.in.w &&
            //             is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
            //         if (valid) {
            //             half2 val = reinterpret_cast<const half2*>(batch_in)[(in_y_im * in_row_vals + in_x_im * dim.in.c + in_c) / 2];
            //             s_in[in_y * w_in + in_x][lane_idx] = val;
            //         } else {
            //             s_in[in_y * w_in + in_x][lane_idx] = __half2half2(__float2half(0.0f));
            //         }
            //     }
            // } else 
            if (dim.in.c == 3) {
                for (int val_idx = threadIdx.x; val_idx < n_in_px * 2; val_idx += BLOCK_SIZE) {
                    const int px_idx = val_idx / 2;
                    const int in_c = (val_idx % 2) * 2;
                    const int in_y = px_idx / w_in; 
                    const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                    const int in_x = px_idx % w_in;
                    const int in_x_im = in_x * DILATION_X + tile_start_in_x; 
                    const bool valid1 = 
                        in_y + tile_start_in_y >= 0 && in_x + tile_start_in_x >= 0 &&
                        in_y + tile_start_in_y < dim.in.h && in_x + tile_start_in_x < dim.in.w &&
                        is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
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
                    s_in[in_y * w_in + in_x][in_c/2] = val;
                }
            } else {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const int in_c = in_c_off + lane_idx * 2;
                    const bool valid1 = 
                        in_y + tile_start_in_y >= 0 && in_x + tile_start_in_x >= 0 &&
                        in_y + tile_start_in_y < dim.in.h && in_x + tile_start_in_x < dim.in.w &&
                        is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask) && in_c < dim.in.c;
                    const bool valid2 = in_c + 1 < dim.in.c;
                    
                    const int in_y_im = in_y * DILATION_Y + tile_start_in_y;
                    const int in_x_im = in_x * DILATION_X + tile_start_in_x; 

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
            

            if (out_c < dim.out.c) {
                for(int in_c = 0; in_c < 64 && in_c + in_c_off < dim.in.c; ++in_c) {
                    const scalar_t *in_c_filter = &filter[(in_c_off+in_c) * KERNEL_SIZE*KERNEL_SIZE * out_c_aligned + out_c];
                    half2 t_f[KERNEL_SIZE*KERNEL_SIZE];
                    #pragma unroll
                    for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                        #pragma unroll
                        for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                            t_f[f_y*KERNEL_SIZE + f_x] = *reinterpret_cast<const half2*>(&in_c_filter[(f_y * KERNEL_SIZE + f_x) * out_c_aligned]);
                        }
                    }
                
                    const int IN_OFF = 0;
                    // in_c % 2 check is used to speedup half2 multiplications without increasing registers much.
                    // if == 0 --> take values as is and multiply them with filter
                    // if != 0 --> swap values and multiply them with filter next filter pair
                    // the filter pairs are pre-processed to match this pattern
                    if (in_c % 2 == 0) {
                        #pragma unroll
                        for (int in_y = IN_OFF; in_y < h_in + IN_OFF; ++in_y) {
                            #pragma unroll
                            for (int in_x = IN_OFF; in_x < w_in + IN_OFF; ++in_x) {
                                const half2 val = s_in[(in_y-IN_OFF) * w_in + (in_x-IN_OFF)][in_c/2];
                                const int min_f_y = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_y)*STRIDE);
                                const int min_f_x = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_x)*STRIDE);
                                const int max_f_y = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockY - (in_y-IN_PIXELS_OFFSET)*STRIDE);
                                const int max_f_x = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockX - (in_x-IN_PIXELS_OFFSET)*STRIDE);
                                #pragma unroll
                                for (int f_y = min_f_y; f_y < max_f_y; ++f_y) {
                                    #pragma unroll
                                    for (int f_x = min_f_x; f_x < max_f_x; ++f_x) {
                                        const int out_y = f_y + (in_y-IN_PIXELS_OFFSET) * STRIDE;
                                        const int out_x = f_x + (in_x-IN_PIXELS_OFFSET) * STRIDE;
                                        const int t_out_idx = out_y * pixelsPerBlockX + out_x;
                                        t_out[t_out_idx] = __hfma2(val, t_f[f_y*KERNEL_SIZE + f_x], t_out[t_out_idx]);
                                    }
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int in_y = IN_OFF; in_y < h_in + IN_OFF; ++in_y) {
                            #pragma unroll
                            for (int in_x = IN_OFF; in_x < w_in + IN_OFF; ++in_x) {
                                const half2 val = __lowhigh2highlow(s_in[(in_y-IN_OFF) * w_in + (in_x-IN_OFF)][in_c/2]);
                                const int min_f_y = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_y)*STRIDE);
                                const int min_f_x = Utils::constexpr_max(0, (IN_PIXELS_OFFSET-in_x)*STRIDE);
                                const int max_f_y = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockY - (in_y-IN_PIXELS_OFFSET)*STRIDE);
                                const int max_f_x = Utils::constexpr_min(KERNEL_SIZE, pixelsPerBlockX - (in_x-IN_PIXELS_OFFSET)*STRIDE);
                                #pragma unroll
                                for (int f_y = min_f_y; f_y < max_f_y; ++f_y) {
                                    #pragma unroll
                                    for (int f_x = min_f_x; f_x < max_f_x; ++f_x) {
                                        const int out_y = f_y + (in_y-IN_PIXELS_OFFSET) * STRIDE;
                                        const int out_x = f_x + (in_x-IN_PIXELS_OFFSET) * STRIDE;
                                        const int t_out_idx = out_y * pixelsPerBlockX + out_x;
                                        t_out[t_out_idx] = __hfma2(val, t_f[f_y*KERNEL_SIZE + f_x], t_out[t_out_idx]);
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
                        const bool valid = out_y_im >= 0 && out_y_im < dim.out.h && out_x_im >= 0 && out_x_im < dim.out.w;
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
                        const bool valid = out_y_im >= 0 && out_y_im < dim.out.h && out_x_im >= 0 && out_x_im < dim.out.w;
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
                    const bool valid =out_y_im >= 0 &&  out_y_im < dim.out.h && out_x_im >= 0 && out_x_im < dim.out.w;
                    if (valid) {
                        batch_out[(out_y_im * dim.out.w + out_x_im) * dim.out.c + out_c] = __low2half(t_out[out_y*pixelsPerBlockX + out_x]);
                    }
                }
            }
        }
    }
}



template<typename scalar_t = float>
void delta_deconv(scalar_t *input, scalar_t *output, scalar_t *filter, scalar_t *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    const bool ENABLE_DILATION = false;
    if (config.stride[0] == 2 && config.stride[1] == 2 && config.kernel_size[0] == 4 && config.kernel_size[1] == 4 && config.groups == 1 && config.dilation[0] == 1 && config.dilation[1] == 1) {
        const int kernel_size = 4;
        const int stride = 2;
        const int pixelsPerBlockX = 8;
        const int pixelsPerBlockY = 8;
        uint32_t y = divup(dim.out.h+config.padding[0], pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
        uint32_t x = divup(dim.out.w+config.padding[1], pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        
        const uint32_t threads = 64;
        const int out_channels_per_block = threads;

        dim3 gridDim(x, y, dim.batch_size * divup(dim.out.c, out_channels_per_block));
        
        delta_deconv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
            input, output, filter, bias, mask, out_mask, dim, config);
    } else if (config.stride[0] == 4 && config.stride[1] == 4 && config.kernel_size[0] == 6 && config.kernel_size[1] == 6 && config.groups == 1 && config.dilation[0] == 1 && config.dilation[1] == 1) {
        const int kernel_size = 6;
        const int stride = 4;
        const int pixelsPerBlockX = 8;
        const int pixelsPerBlockY = 8;
        
        uint32_t y = divup(dim.out.h+config.padding[0], pixelsPerBlockY);
        uint32_t x = divup(dim.out.w+config.padding[1], pixelsPerBlockX);

        const uint32_t threads = 64;
        const int out_channels_per_block = threads;

        dim3 gridDim(x, y, dim.batch_size * divup(dim.out.c, out_channels_per_block));
        
        delta_deconv_sp<scalar_t, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, false, stride, ENABLE_DILATION><<<gridDim, threads>>>(
            input, output, filter, bias, mask, out_mask, dim, config);
    }
     else {
        printf("Transposed convolution only supports 4x4/6x6 kernels with groups=1 and stride=2/4 and dilation=1. Got k=%ix%i g=%i s=%ix%i\n", config.kernel_size[0], config.kernel_size[1], config.groups, config.stride[0], config.stride[1]);
        throw "Transposed convolution only supports 4x4/6x6 kernels with groups=1 and stride=2/4 and dilation=1";
    }
}



void delta_deconv_hp(half *input, half *output, half *filter, half *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config) {
    const bool ENABLE_DILATION = false;
    if (config.stride[0] == 2 && config.stride[1] == 2 && config.kernel_size[0] == 4 && config.kernel_size[1] == 4 && config.groups == 1 && config.dilation[0] == 1 && config.dilation[1] == 1) {
        const int kernel_size = 4;
        const int stride = 2;
        const int pixelsPerBlockX = 4;
        const int pixelsPerBlockY = 4;
        uint32_t y = divup(dim.out.h+config.padding[0], pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
        uint32_t x = divup(dim.out.w+config.padding[1], pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        
        dim3 gridDim(x, y, dim.batch_size);
        
        const uint32_t threads = 64;
        const int out_channels_per_block = threads;
        delta_deconv_hp_kernel<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
            input, output, filter, bias, mask, out_mask, dim, config);
    } else if (config.stride[0] == 4 && config.stride[1] == 4 && config.kernel_size[0] == 6 && config.kernel_size[1] == 6 && config.groups == 1 && config.dilation[0] == 1 && config.dilation[1] == 1) {
        const int kernel_size = 6;
        const int stride = 4;
        const int pixelsPerBlockX = 4;
        const int pixelsPerBlockY = 4;
        uint32_t y = divup(dim.out.h+config.padding[0], pixelsPerBlockY * config.dilation[0]) * config.dilation[0];
        uint32_t x = divup(dim.out.w+config.padding[1], pixelsPerBlockX * config.dilation[1]) * config.dilation[1];
        
        dim3 gridDim(x, y, dim.batch_size);
        
        const uint32_t threads = 64;
        const int out_channels_per_block = threads;
        delta_deconv_hp_kernel<half, kernel_size, pixelsPerBlockX, pixelsPerBlockY, threads, out_channels_per_block, true, true, stride, ENABLE_DILATION><<<gridDim, threads>>>(
            input, output, filter, bias, mask, out_mask, dim, config);
    }
     else {
        printf("Transposed convolution only supports 4x4/6x6 kernels with groups=1 and stride=2/4 and dilation=1. Got k=%ix%i g=%i s=%ix%i\n", config.kernel_size[0], config.kernel_size[1], config.groups, config.stride[0], config.stride[1]);
        throw "Transposed convolution only supports 4x4/6x6 kernels with groups=1 and stride=2/4 and dilation=1";
    }
}


template void delta_deconv<float>(float *input, float *output, float *filter, float *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config);
