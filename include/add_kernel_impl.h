#pragma once

#include "device_common.h"

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

template <typename T>
__global__ void gpu_add_kernel(const T *a, const T *b, T *out, size_t n) {
    constexpr int vec_size = 16 / sizeof(T);
    int block_work_size = blockDim.x * vec_size;
    size_t index = blockIdx.x * block_work_size + threadIdx.x * vec_size;
    int remaining = n - index;
    if (remaining < vec_size) {
        for (auto i = index; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    } else {
        using vec_t = aligned_array<T, vec_size>;
        auto a_vec = *reinterpret_cast<vec_t *>(const_cast<T *>(&a[index]));
        auto b_vec = *reinterpret_cast<vec_t *>(const_cast<T *>(&b[index]));
        auto out_vec_ptr = reinterpret_cast<vec_t *>(&out[index]);
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            a_vec.val[i] += b_vec.val[i];
        }
        *out_vec_ptr = a_vec;
    }
}

template <typename T>
void gpu_add(const T *a, const T *b, T *out, size_t n) {
    constexpr int vec_size = 16 / sizeof(T);
    constexpr int block_size = 256;
    constexpr int block_work_size = block_size * vec_size;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((n + block_work_size - 1) / block_work_size);
    gpu_add_kernel<T><<<numBlocks, threadsPerBlock>>>(a, b, out, n);
    gpuDeviceSynchronize();
}
