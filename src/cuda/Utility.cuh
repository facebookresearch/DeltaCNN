// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <stdio.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "common.cuh"

static constexpr unsigned int FULL_MASK{0xFFFFFFFF};
static constexpr unsigned int WARP_SIZE{32};

void inline start_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&end));
	HANDLE_ERROR(cudaEventRecord(start, 0));
}

// ##############################################################################################################################################
//
float inline end_clock(cudaEvent_t &start, cudaEvent_t &end)
{
	float time;
	HANDLE_ERROR(cudaEventRecord(end, 0));
	HANDLE_ERROR(cudaEventSynchronize(end));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(end));

	// Returns ms
	return time;
}

namespace Utils
{

    template <typename T>
    constexpr T constexpr_min(const T a, const T b) {
        return a > b ? b : a;
    }
    template <typename T>
    constexpr T constexpr_max(const T a, const T b) {
        return a < b ? b : a;
    }
    // ##############################################################################################################################################
    //
    __device__ __forceinline__ int lane_id()
    {
        return (threadIdx.x & 31);
    }
    
    // ##############################################################################################################################################
    //
    __device__ __forceinline__ int warp_id()
    {
        return (threadIdx.x >> 5);
    }
    
    // ##############################################################################################################################################
    //
    template<typename T, typename T2>
    __host__ __device__ __forceinline__ T divup(T a, T2 b)
    {
        return (a + b - 1) / b;
    }

    // ##############################################################################################################################################
    //
	template <typename T>
	static __device__ __forceinline__ int getNextPow2Pow(T n)
	{
		if ((n & (n - 1)) == 0)
			return 32 - __clz(n) - 1;
		else
			return 32 - __clz(n);
    }

    // ##############################################################################################################################################
    //
    template <typename T>
	static __device__ __forceinline__ T getNextPow2(T n)
	{
		return 1 << (getNextPow2Pow(n));
	}
    
    // ##############################################################################################################################################
    //
    template<unsigned int X, int Completed = 0>
	struct static_clz
	{
		static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
	};

	// ##############################################################################################################################################
	//
	template<unsigned int X>
	struct static_clz<X, 32>
	{
		static const int value = 32;
	};

    // ##############################################################################################################################################
    //
	template <int32_t n>
	static constexpr int static_getNextPow2Pow()
	{
		if ((n & (n - 1)) == 0)
			return 32 - static_clz<static_cast<unsigned int>(n)>::value - 1;
		else
			return 32 - static_clz<static_cast<unsigned int>(n)>::value;
	}

    // ##############################################################################################################################################
    //
	template <int32_t n>
	static constexpr size_t static_getNextPow2()
	{
		return 1 << (static_getNextPow2Pow<n>());
	}
    
    // ##############################################################################################################################################
    //
    template <typename T>
    static constexpr bool isPowerOfTwo(const T n) 
    {
        return (n & (n - 1)) == 0;
    }
    
    // ##############################################################################################################################################
    //
    template <int32_t size>
    static constexpr __forceinline__ __device__ int32_t modPower2(const int32_t value)
    {
        static_assert(isPowerOfTwo(size), "ModPower2 used with non-power of 2");
        return value & (size - 1);
    }
    
    // ##############################################################################################################################################
    //
    template <int32_t divisor, typename T>
    __host__ __device__ __forceinline__ T divPower2(T val)
    {
        return val >> static_getNextPow2Pow<divisor>();
    }

    // ##############################################################################################################################################
    //
    template <int32_t multiplicator, typename T>
    __host__ __device__ __forceinline__ T mulPower2(T val)
    {
        return val << static_getNextPow2Pow<multiplicator>();
    }
    
    // ##############################################################################################################################################
    //
    template<typename T>
    __device__ __forceinline__ T warpReduceSum(T val, unsigned mask = FULL_MASK) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        return val;
    }
    
    // ##############################################################################################################################################
    //
    template<> __device__ __forceinline__ half2 warpReduceSum(half2 val, unsigned mask) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            int tmp_val = __shfl_down_sync(mask, *reinterpret_cast<int32_t*>(&val), offset);
            val = __hadd2(val, *reinterpret_cast<half2*>(&tmp_val));
        }
        return val;
    }
    
    // ##############################################################################################################################################
    //
    template<> __device__ __forceinline__ half warpReduceSum(half val, unsigned mask) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float tmp_val = __shfl_down_sync(mask, __half2float(val), offset);
            val = __hadd(val, __float2half(tmp_val));
        }
        return val;
    }
    
    // ##############################################################################################################################################
    //
    template<typename T>
    __device__ __forceinline__ T warpReduceSumGrouped(T val, int groups, unsigned mask = FULL_MASK) {
        for (int offset = WARP_SIZE / 2 / groups; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        return val;
    }
    
    // ##############################################################################################################################################
    //
    template<typename T>
    __device__ __forceinline__ T warpReduceMax(T val, unsigned mask = FULL_MASK) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val = max(__shfl_down_sync(mask, val, offset), val);
        }
        return val;
    }
    
    // ##############################################################################################################################################
    //
    __device__ __forceinline__ static float atomicMax(float* address, float val)
    {
        int* address_as_i = (int*) address;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
}

