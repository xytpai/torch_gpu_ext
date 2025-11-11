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
    const int nw = blockDim.x / 32;
    const int nw_packed = nw / pack_size;
    const int wid_packed = wid % nw_packed;
    const int blockdim_packed = blockDim.x / pack_size;

    if (nw_packed <= 0) {
        shared[threadIdx.x] = val;
        __syncthreads();
        int start = pack_id * blockdim_packed;
        int end = start + blockdim_packed;
        end = end < blockDim.x ? end : blockDim.x;
        T acc = 0;
        for (int i = start; i < end; ++i) {
            acc += shared[i];
        }
        return acc;
    } else {
        val = warp_reduce_sum(val);
        if (w_tid == 0) {
            shared[wid] = val;
        }
        __syncthreads();
        int w_start = pack_id * nw_packed;
        int w_end = w_start + nw_packed;
        w_end = w_end < nw ? w_end : nw;
        T acc = 0;
        for (int i = w_start; i < w_end; ++i) {
            acc += shared[i];
        }
        return acc;
    }
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

template <typename T, int VEC_SIZE, int PACK>
__device__ __forceinline__ void rms_norm_(
    vec_t<T, VEC_SIZE> (&input)[PACK],
    vec_t<T, VEC_SIZE> (&gamma)[PACK],
    float rms_dim,
    float rms_eps,
    int pack_id,
    int pack_size) {
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
    acc = block_utils::block_reduce_sum<float>(acc, pack_id, pack_size);
    auto s_val = rsqrtf(acc / rms_dim + rms_eps);
    __syncthreads();
#pragma unroll
    for (int p = 0; p < PACK; ++p) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            input[p][i] = static_cast<T>((float)input[p][i] * s_val * (float)gamma[p][i]);
        }
    }
}

static constexpr int kBytesPerAccess = 16;

template <typename T>
__global__ void fused_rope_rms_neox_kernel(
    const T *q, const T *k, const T *q_w, const T *k_w, const T *cos_sin,
    T *out_q, T *out_k, int num_tokens, int num_heads, int head_size,
    float eps) {
    constexpr int VEC_SIZE = kBytesPerAccess / sizeof(T);
    int numel = num_tokens * num_heads * head_size;
    int half_head_size = head_size / 2;

    int pack_size = (blockDim.x * VEC_SIZE) / half_head_size;
    int pack_id = (threadIdx.x * VEC_SIZE) / half_head_size;
    int global_head_id = blockIdx.x * pack_size + pack_id;
    int token_id = global_head_id / num_heads;
    int access_id_in_head = (threadIdx.x * VEC_SIZE) % half_head_size;

    vec_t<T, VEC_SIZE> q_w_vec[2], k_w_vec[2];
    q_w_vec[0].load(q_w + access_id_in_head);
    q_w_vec[1].load(q_w + access_id_in_head + half_head_size);
    k_w_vec[0].load(k_w + access_id_in_head);
    k_w_vec[1].load(k_w + access_id_in_head + half_head_size);
    for (int idx = global_head_id * head_size + access_id_in_head; idx < numel; idx += gridDim.x * pack_size * head_size) {
        vec_t<T, VEC_SIZE> q_vec[2], k_vec[2], cos_vec, sin_vec;
        q_vec[0].load(q + idx);
        q_vec[1].load(q + idx + half_head_size);
        k_vec[0].load(k + idx);
        k_vec[1].load(k + idx + half_head_size);
        cos_vec.load(&cos_sin[token_id * head_size + access_id_in_head]);
        sin_vec.load(&cos_sin[token_id * head_size + access_id_in_head + half_head_size]);
        rms_norm_<T, VEC_SIZE, 2>(q_vec, q_w_vec, head_size, eps, pack_id, pack_size);
        rms_norm_<T, VEC_SIZE, 2>(k_vec, k_w_vec, head_size, eps, pack_id, pack_size);
        vec_t<T, VEC_SIZE> oq_vec[2], ok_vec[2];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            oq_vec[0][i] = q_vec[0][i] * cos_vec[i] - q_vec[1][i] * sin_vec[i];
            oq_vec[1][i] = q_vec[1][i] * cos_vec[i] + q_vec[0][i] * sin_vec[i];
            ok_vec[0][i] = k_vec[0][i] * cos_vec[i] - k_vec[1][i] * sin_vec[i];
            ok_vec[1][i] = k_vec[1][i] * cos_vec[i] + k_vec[0][i] * sin_vec[i];
        }
        oq_vec[0].store(out_q + idx);
        oq_vec[1].store(out_q + idx + half_head_size);
        ok_vec[0].store(out_k + idx);
        ok_vec[1].store(out_k + idx + half_head_size);
    }
}

template <typename T>
__global__ void fused_rope_rms_noneox_kernel(
    const T *q, const T *k, const T *q_w, const T *k_w, const T *cos_sin,
    T *out_q, T *out_k, int num_tokens, int num_heads, int head_size,
    float eps) {
    constexpr int VEC_SIZE = kBytesPerAccess / sizeof(T);
    int numel = num_tokens * num_heads * head_size;
    int half_head_size = head_size / 2;

    int pack_size = (blockDim.x * VEC_SIZE) / head_size;
    int pack_id = (threadIdx.x * VEC_SIZE) / head_size;
    int global_head_id = blockIdx.x * pack_size + pack_id;
    int token_id = global_head_id / num_heads;
    int access_id_in_head = (threadIdx.x * VEC_SIZE) % head_size;

    vec_t<T, VEC_SIZE> q_w_vec[1], k_w_vec[1];
    q_w_vec[0].load(q_w + access_id_in_head);
    k_w_vec[0].load(k_w + access_id_in_head);
    for (int idx = global_head_id * head_size + access_id_in_head; idx < numel; idx += gridDim.x * pack_size * head_size) {
        vec_t<T, VEC_SIZE> q_vec[1], k_vec[1];
        vec_t<T, VEC_SIZE / 2> cos_vec[1], sin_vec[1];
        q_vec[0].load(q + idx);
        k_vec[0].load(k + idx);
        cos_vec[0].load(&cos_sin[token_id * head_size + access_id_in_head / 2]);
        sin_vec[0].load(&cos_sin[token_id * head_size + access_id_in_head / 2 + half_head_size]);
        rms_norm_<T, VEC_SIZE, 1>(q_vec, q_w_vec, head_size, eps, pack_id, pack_size);
        rms_norm_<T, VEC_SIZE, 1>(k_vec, k_w_vec, head_size, eps, pack_id, pack_size);
        vec_t<T, VEC_SIZE> oq_vec[2], ok_vec[2];
#pragma unroll
        for (int i = 0; i < VEC_SIZE / 2; ++i) {
            oq_vec[0][2 * i + 0] = q_vec[0][2 * i + 0] * cos_vec[0][i] - q_vec[0][2 * i + 1] * sin_vec[0][i];
            oq_vec[0][2 * i + 1] = q_vec[0][2 * i + 1] * cos_vec[0][i] + q_vec[0][2 * i + 0] * sin_vec[0][i];
            ok_vec[0][2 * i + 0] = k_vec[0][2 * i + 0] * cos_vec[0][i] - k_vec[0][2 * i + 1] * sin_vec[0][i];
            ok_vec[0][2 * i + 1] = k_vec[0][2 * i + 1] * cos_vec[0][i] + k_vec[0][2 * i + 0] * sin_vec[0][i];
        }
        oq_vec[0].store(out_q + idx);
        ok_vec[0].store(out_k + idx);
    }
}

template <typename T>
void fused_rope_rms(
    const T *q, const T *k, const T *q_w, const T *k_w, const T *cos_sin,
    T *out_q, T *out_k,
    int num_tokens, int num_heads, int head_size,
    bool is_neox_style, float eps, gpuStream_t stream) {
    constexpr int VEC_SIZE = kBytesPerAccess / sizeof(T);
    assert(head_size % 32 == 0 && head_size >= 32);
    int pack_size;
    switch (head_size) {
    case 256:
        pack_size = 4;
        break;
    case 128:
        pack_size = 8;
        break;
    case 64:
        pack_size = 16;
        break;
    default:
        pack_size = 1;
        break;
    }
    if (is_neox_style) {
        dim3 threadsPerBlock(head_size / VEC_SIZE / 2 * pack_size);
        dim3 numBlocks(num_tokens * num_heads / pack_size);
        fused_rope_rms_neox_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
            q, k, q_w, k_w, cos_sin, out_q, out_k, num_tokens, num_heads, head_size, eps);
    } else {
        pack_size = pack_size == 1 ? 1 : pack_size / 2;
        dim3 threadsPerBlock(head_size / VEC_SIZE * pack_size);
        dim3 numBlocks(num_tokens * num_heads / pack_size);
        fused_rope_rms_noneox_kernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
            q, k, q_w, k_w, cos_sin, out_q, out_k, num_tokens, num_heads, head_size, eps);
    }
}

} // namespace rope_rms
