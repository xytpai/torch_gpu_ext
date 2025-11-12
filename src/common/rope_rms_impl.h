#include "gpu_loops_kernel_impl.h"
#include "device_common.h"

namespace rope_rms {

static constexpr int kBytesPerAccess = 16;

namespace block_utils {

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
#ifdef __CUDACC__
        val += __shfl_xor_sync(0xffffffff, val, offset, 32);
#else
        val += __shfl_xor(val, offset, 32);
#endif
    return val;
}

template <typename T>
__inline__ __device__ T warp_broadcast(T val) {
#ifdef __CUDACC__
    return __shfl_sync(0xffffffff, val, 0);
#else
    return __shfl_sync(__activemask(), val, 0, 32);
#endif
}

} // namespace block_utils

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) vec_t {
    T data[vec_size];
    __device__ __forceinline__ T &operator[](int i) {
        return data[i];
    }
    __device__ __forceinline__ T const &operator[](int i) const {
        return data[i];
    }
    __device__ __forceinline__ void load(const T *ptr) {
        *this = *reinterpret_cast<vec_t<T, vec_size> *>(const_cast<T *>(ptr));
    }
    __device__ __forceinline__ void loop_load(const T *ptr) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = ptr[i];
        }
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, vec_size> *>(ptr) = *this;
    }
    __device__ __forceinline__ void loop_store(T *ptr) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            ptr[i] = data[i];
        }
    }
    __device__ __forceinline__ void nontemporal_load(const T *ptr) {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint64_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            *reinterpret_cast<uint64_t *>((char *)data + i * sizeof(uint64_t)) =
                __builtin_nontemporal_load((uint64_t *)((char *)ptr + i * sizeof(uint64_t)));
        }
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint64_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __builtin_nontemporal_store(*reinterpret_cast<uint64_t *>((char *)data + i * sizeof(uint64_t)),
                                        (uint64_t *)((char *)ptr + i * sizeof(uint64_t)));
        }
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = val;
        }
    }
};

template <typename T, int VEC_SIZE, int PACK>
__device__ __forceinline__ void warp_rms_norm_(
    vec_t<T, VEC_SIZE> (&input)[PACK],
    vec_t<T, VEC_SIZE> (&gamma)[PACK],
    float rms_dim,
    float rms_eps) {
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int p = 0; p < PACK; ++p) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            float v = (float)input[p][i];
            acc += v * v;
        }
    }
    int warp_id = threadIdx.x / 32;
    int warp_t_id = threadIdx.x % 32;
    acc = block_utils::warp_reduce_sum<float>(acc);
    acc = block_utils::warp_broadcast(acc);
    __syncwarp();
    auto s_val = rsqrtf(acc / rms_dim + rms_eps);
#pragma unroll
    for (int p = 0; p < PACK; ++p) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            input[p][i] = static_cast<T>((float)input[p][i] * s_val * (float)gamma[p][i]);
        }
    }
}

template <typename T, int HEAD_SIZE>
__global__ void fused_rope_rms_neox_kernel(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, double eps, int64_t total_warps) {
    constexpr int VEC_SIZE = HEAD_SIZE / 32 / 2;
    constexpr int HALF_HEAD_SIZE = HEAD_SIZE / 2;
    const auto warp_id = threadIdx.x / 32;
    const auto num_warps_per_block = blockDim.x / 32;
    const auto global_warp_id = blockIdx.x * num_warps_per_block + warp_id;
    if (global_warp_id >= total_warps) {
        return;
    }
    auto token_id = global_warp_id / (num_heads_q + num_heads_k);
    auto head_id_in_token = global_warp_id % (num_heads_q + num_heads_k);
    bool is_q = head_id_in_token < num_heads_q;
    auto access_id_in_head = (threadIdx.x % 32) * VEC_SIZE;
    auto qkv_ = qkv + token_id * (num_heads_q + num_heads_k + num_heads_v) * HEAD_SIZE + head_id_in_token * HEAD_SIZE;
    auto position_ = positions[token_id];

    vec_t<T, VEC_SIZE> w_vec[2];

    if (is_q) {
        w_vec[0].nontemporal_load(q_w + access_id_in_head);
        w_vec[1].nontemporal_load(q_w + access_id_in_head + HALF_HEAD_SIZE);
    } else {
        w_vec[0].nontemporal_load(k_w + access_id_in_head);
        w_vec[1].nontemporal_load(k_w + access_id_in_head + HALF_HEAD_SIZE);
    }

    vec_t<T, VEC_SIZE> x_vec[2], cos_vec, sin_vec;
    x_vec[0].load(qkv_ + access_id_in_head);
    x_vec[1].load(qkv_ + access_id_in_head + HALF_HEAD_SIZE);
    cos_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head]);
    sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head + HALF_HEAD_SIZE]);

    warp_rms_norm_<T, VEC_SIZE, 2>(x_vec, w_vec, HEAD_SIZE, eps);

    vec_t<T, VEC_SIZE> out_vec[2];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        out_vec[0][i] = x_vec[0][i] * cos_vec[i] - x_vec[1][i] * sin_vec[i];
        out_vec[1][i] = x_vec[1][i] * cos_vec[i] + x_vec[0][i] * sin_vec[i];
    }

    out_vec[0].store(qkv_ + access_id_in_head);
    out_vec[1].store(qkv_ + access_id_in_head + HALF_HEAD_SIZE);
}

template <typename T, int HEAD_SIZE>
__global__ void fused_rope_rms_noneox_kernel(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, double eps, int64_t total_warps) {
    constexpr int VEC_SIZE = HEAD_SIZE / 32;
    constexpr int HALF_HEAD_SIZE = HEAD_SIZE / 2;
    const auto warp_id = threadIdx.x / 32;
    const auto num_warps_per_block = blockDim.x / 32;
    const auto global_warp_id = blockIdx.x * num_warps_per_block + warp_id;
    if (global_warp_id >= total_warps) {
        return;
    }
    auto token_id = global_warp_id / (num_heads_q + num_heads_k);
    auto head_id_in_token = global_warp_id % (num_heads_q + num_heads_k);
    bool is_q = head_id_in_token < num_heads_q;
    auto access_id_in_head = (threadIdx.x % 32) * VEC_SIZE;
    auto qkv_ = qkv + token_id * (num_heads_q + num_heads_k + num_heads_v) * HEAD_SIZE + head_id_in_token * HEAD_SIZE;
    auto position_ = positions[token_id];

    vec_t<T, VEC_SIZE> w_vec[1];

    if (is_q) {
        w_vec[0].nontemporal_load(q_w + access_id_in_head);
    } else {
        w_vec[0].nontemporal_load(k_w + access_id_in_head);
    }

    vec_t<T, VEC_SIZE> x_vec[1], cos_vec, sin_vec;
    x_vec[0].load(qkv_ + access_id_in_head);
    cos_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2]);
    sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);

    warp_rms_norm_<T, VEC_SIZE, 1>(x_vec, w_vec, HEAD_SIZE, eps);

    vec_t<T, VEC_SIZE> out_vec[1];
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
        out_vec[0][2 * i + 0] = x_vec[0][2 * i + 0] * cos_vec[i] - x_vec[0][2 * i + 1] * sin_vec[i];
        out_vec[0][2 * i + 1] = x_vec[0][2 * i + 1] * cos_vec[i] + x_vec[0][2 * i + 0] * sin_vec[i];
    }

    out_vec[0].store(qkv_ + access_id_in_head);
}

template <typename T>
void fused_rope_rms(
    T *qkv, const T *q_w, const T *k_w, const T *cos_sin, const int64_t *positions,
    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
    bool is_neox_style, double eps, gpuStream_t stream) {
    assert(head_size == 64 || head_size == 128 || head_size == 256);
    constexpr int block_size = 256;
    auto total_warps = num_tokens * (num_heads_q + num_heads_k);
    auto num_warps_per_block = block_size / 32;

#define DISPATCH_NEOX(HEAD_SIZE)                                                                         \
    if (is_neox_style) {                                                                                 \
        fused_rope_rms_neox_kernel<T, HEAD_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(             \
            qkv, q_w, k_w, cos_sin, positions, num_heads_q, num_heads_k, num_heads_v, eps, total_warps); \
    } else {                                                                                             \
        fused_rope_rms_noneox_kernel<T, HEAD_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(           \
            qkv, q_w, k_w, cos_sin, positions, num_heads_q, num_heads_k, num_heads_v, eps, total_warps); \
    }

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks((total_warps + num_warps_per_block - 1) / num_warps_per_block);
    switch (head_size) {
    case 64:
        DISPATCH_NEOX(64)
        break;
    case 128:
        DISPATCH_NEOX(128)
        break;
    case 256:
        DISPATCH_NEOX(256)
        break;
    }

#undef DISPATCH_NEOX
}

} // namespace rope_rms
