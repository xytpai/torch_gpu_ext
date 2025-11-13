#include "gpu_loops_kernel_impl.h"
#include "device_common.h"

namespace rope_rms {

static constexpr int kBytesPerAccess = 16;

namespace block_utils {

template <typename T>
__inline__ __device__ T warp_shfl_xor_sync(T val, int offset) {
#ifdef __CUDACC__
    return __shfl_xor_sync(0xffffffff, val, offset, 32);
#else
    return __shfl_xor(val, offset, 32);
#endif
}

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += warp_shfl_xor_sync(val, offset);
    return val;
}

template <typename T>
__inline__ __device__ T warp_shfl_sync(T val, int src_id) {
#ifdef __CUDACC__
    return __shfl_sync(0xffffffff, val, src_id);
#else
    return __shfl_sync(__activemask(), val, src_id, 32);
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
            reinterpret_cast<uint64_t *>(&data)[i] = __builtin_nontemporal_load((uint64_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint64_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __builtin_nontemporal_store(reinterpret_cast<uint64_t *>(&data)[i], (uint64_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = val;
        }
    }
};

template <typename T, int vec_size>
__inline__ __device__ vec_t<T, vec_size> warp_shfl_sync_vec(vec_t<T, vec_size> &val, int offset) {
    constexpr int ITERS = vec_size * sizeof(T) / sizeof(uint64_t);
    vec_t<T, vec_size> out;
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        uint64_t val_ = reinterpret_cast<uint64_t *>(&val)[i];
        reinterpret_cast<uint64_t *>(&out)[i] = block_utils::warp_shfl_sync<uint64_t>(val_, offset);
    }
    return out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void warp_rms_norm_(
    vec_t<T, VEC_SIZE> &input,
    vec_t<T, VEC_SIZE> &gamma,
    float rms_dim,
    float rms_eps) {
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = (float)input[i];
        acc += v * v;
    }
    int warp_id = threadIdx.x / 32;
    int warp_t_id = threadIdx.x % 32;
    acc = block_utils::warp_reduce_sum<float>(acc);
    acc = block_utils::warp_shfl_sync(acc, 0);
    __syncwarp();
    auto s_val = rsqrtf(acc / rms_dim + rms_eps);
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        input[i] = static_cast<T>((float)input[i] * s_val * (float)gamma[i]);
    }
}

template <typename T, int HEAD_SIZE>
__global__ void fused_rope_rms_neox_kernel(
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
    auto neighbor_offset = access_id_in_head < HALF_HEAD_SIZE ? HALF_HEAD_SIZE / VEC_SIZE : -HALF_HEAD_SIZE / VEC_SIZE;
    auto qkv_ = qkv + token_id * (num_heads_q + num_heads_k + num_heads_v) * HEAD_SIZE + head_id_in_token * HEAD_SIZE;
    auto position_ = positions[token_id];

    vec_t<T, VEC_SIZE> w_vec;

    if (is_q) {
        w_vec.nontemporal_load(q_w + access_id_in_head);
    } else {
        w_vec.nontemporal_load(k_w + access_id_in_head);
    }

    vec_t<T, VEC_SIZE> x_vec, cos_sin_vec;
    x_vec.load(qkv_ + access_id_in_head);
    cos_sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head]);
    warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);
    auto nb_cos_sin_vec = warp_shfl_sync_vec<T, VEC_SIZE>(cos_sin_vec, threadIdx.x + neighbor_offset);
    auto nb_x_vec = warp_shfl_sync_vec<T, VEC_SIZE>(x_vec, threadIdx.x + neighbor_offset);
    vec_t<T, VEC_SIZE> out_vec;
    if (neighbor_offset > 0) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vec[i] = x_vec[i] * cos_sin_vec[i] - nb_x_vec[i] * nb_cos_sin_vec[i]; // x0 * cos - x1 * sin
        }
    } else {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            out_vec[i] = x_vec[i] * nb_cos_sin_vec[i] + nb_x_vec[i] * cos_sin_vec[i]; // x1 * cos + x0 * sin
        }
    }
    out_vec.store(qkv_ + access_id_in_head);
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

    vec_t<T, VEC_SIZE> w_vec;

    if (is_q) {
        w_vec.nontemporal_load(q_w + access_id_in_head);
    } else {
        w_vec.nontemporal_load(k_w + access_id_in_head);
    }

    vec_t<T, VEC_SIZE> x_vec, cos_vec, sin_vec;
    x_vec.load(qkv_ + access_id_in_head);
    cos_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2]);
    sin_vec.load(&cos_sin[position_ * HEAD_SIZE + access_id_in_head / 2 + HALF_HEAD_SIZE]);

    warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, HEAD_SIZE, eps);

    vec_t<T, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE / 2; ++i) {
        out_vec[2 * i + 0] = x_vec[2 * i + 0] * cos_vec[i] - x_vec[2 * i + 1] * sin_vec[i];
        out_vec[2 * i + 1] = x_vec[2 * i + 1] * cos_vec[i] + x_vec[2 * i + 0] * sin_vec[i];
    }

    out_vec.store(qkv_ + access_id_in_head);
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
