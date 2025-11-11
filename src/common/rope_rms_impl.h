#include "gpu_loops_kernel_impl.h"
#include "device_common.h"

namespace rope_rms {

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
__inline__ __device__ T block_reduce_sum(T val, int pack_id, int pack_size) {
    static __shared__ T shared[32];
    const int tid = threadIdx.x;
    const int w_tid = tid % 32;
    const int wid = tid / 32;
    val = warp_reduce_sum(val);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    int nw = blockDim.x / 32;
    int nw_packed = nw / pack_size;
    val = (w_tid < nw_packed) ? shared[wid * nw_packed + w_tid] : (T)(0.f);
    __syncthreads();
    val = warp_reduce_sum(val);
    return val;
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
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, vec_size> *>(ptr) = *this;
    }
    __device__ __forceinline__ void nontemporal_load(const T *ptr) {
        *reinterpret_cast<uint64_t *>(&data[0]) = __builtin_nontemporal_load((uint64_t *)(const_cast<T *>(ptr)));
        *reinterpret_cast<uint64_t *>(&data[vec_size / 2]) = __builtin_nontemporal_load((uint64_t *)((char *)const_cast<T *>(ptr) + 8));
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        __builtin_nontemporal_store(*reinterpret_cast<uint64_t *>(&data[0]), (uint64_t *)ptr);
        __builtin_nontemporal_store(*reinterpret_cast<uint64_t *>(&data[vec_size / 2]), (uint64_t *)((char *)ptr + 8));
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            data[i] = val;
        }
    }
};

template <typename T, int VEC_SIZE>
__device__ __forceinline__ vec_t<T, VEC_SIZE> packed_rms_norm(
    vec_t<T, VEC_SIZE> const &input,
    vec_t<T, VEC_SIZE> const &gamma,
    float rms_dim,
    float rms_eps,
    int pack_id,
    int pack_size) {
    __shared__ float s_val[32];
    vec_t<T, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&input)[i]);
        acc += v * v;
    }
    acc = block_utils::block_reduce_sum<float>(acc, pack_id, pack_size);
    if (threadIdx.x % 32 == 0) {
        s_val[threadIdx.x / 32] = rsqrtf(acc / rms_dim + rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        norm_out.data[i] =
            static_cast<T>(static_cast<float>(reinterpret_cast<T const *>(&input)[i]) * s_val[pack_id] * static_cast<float>(reinterpret_cast<T const *>(&gamma)[i]));
    }
    return norm_out;
}

template <typename T>
__global__ void fused_rope_rms_kernel(
    const T *q, const T *k, const T *q_w, const T *k_w, const T *cos_sin,
    T *out_q, T *out_k, int num_tokens, int num_heads, int head_size,
    int rotary_dim, bool is_neox_style, float eps) {
    constexpr int VEC_SIZE = 16 / sizeof(T);
    int pack_size = (blockDim.x * VEC_SIZE) / head_size;
    int pack_id = (threadIdx.x * VEC_SIZE) / head_size;
    int global_head_id = blockIdx.x * pack_size + pack_id;
    int access_id_in_head = (threadIdx.x * VEC_SIZE) % head_size;
    // int seq_id = blockIdx.x / num_heads;
    // int head_id = blockIdx.x % num_heads;
    int numel = num_tokens * num_heads * head_size;
    vec_t<T, VEC_SIZE> q_w_vec, k_w_vec;
    q_w_vec.nontemporal_load(q_w + access_id_in_head);
    k_w_vec.nontemporal_load(k_w + access_id_in_head);
    for (int idx = global_head_id * head_size + access_id_in_head; idx < numel; idx += gridDim.x * pack_size * head_size) {
        vec_t<T, VEC_SIZE> q_vec, k_vec;
        q_vec.load(q + idx);
        k_vec.load(k + idx);
        auto q_rms_vec = packed_rms_norm(q_vec, q_w_vec, head_size, eps, pack_id, pack_size);
        auto k_rms_vec = packed_rms_norm(k_vec, k_w_vec, head_size, eps, pack_id, pack_size);
        if (access_id_in_head >= rotary_dim) {
            q_rms_vec.store(out_q + idx);
            k_rms_vec.store(out_k + idx);
        }
    }
}

template <typename T>
void fused_rope_rms(
    const T *q, const T *k, const T *q_w, const T *k_w, const T *cos_sin,
    T *out_q, T *out_k,
    int num_tokens, int num_heads, int head_size, int rotary_dim,
    bool is_neox_style, float eps, gpuStream_t stream) {
    constexpr int VEC_SIZE = 16 / sizeof(T);
    assert(head_size % 32 == 0 && head_size >= 32);
    int pack_size;
    switch (head_size) {
    case 128:
        pack_size = 2;
        break;
    case 64:
        pack_size = 4;
        break;
    default:
        pack_size = 1;
        break;
    }
    dim3 threadsPerBlock(head_size / VEC_SIZE * pack_size);
    dim3 numBlocks(num_tokens * num_heads / pack_size);
    fused_rope_rms_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        q, k, q_w, k_w, cos_sin, out_q, out_k, num_tokens, num_heads, head_size,
        rotary_dim, is_neox_style, eps);
}

} // namespace rope_rms
