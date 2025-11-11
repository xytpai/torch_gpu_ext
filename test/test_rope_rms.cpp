#include "rope_rms_impl.h"

namespace test {

template <typename T>
class CPUInputs {
public:
    int num_tokens;
    int num_heads;
    int head_size;
    int rotary_dim;
    bool is_neox_style;
    float eps;
    int numel;
    T *q;
    T *k;
    T *q_w;
    T *k_w;
    T *cos_sin; // num_tokens, head_size
    T *out_q;
    T *out_k;

    CPUInputs(int num_tokens, int num_heads, int head_size, int rotary_dim, bool is_neox_style, float eps) :
        num_tokens(num_tokens), num_heads(num_heads), head_size(head_size),
        rotary_dim(rotary_dim), is_neox_style(is_neox_style), eps(eps) {
        numel = num_tokens * num_heads * head_size;
    }

    void allocate() {
        q = new T[numel];
        k = new T[numel];
        q_w = new T[head_size];
        k_w = new T[head_size];
        cos_sin = new T[num_tokens * head_size];
        out_q = new T[numel];
        out_k = new T[numel];
    }

    void reset() {
        for (int i = 0; i < numel; ++i) {
            q[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
            k[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
            out_q[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
            out_k[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < head_size; ++i) {
            q_w[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
            k_w[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
        for (int i = 0; i < num_tokens * head_size; ++i) {
            cos_sin[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
    }

    ~CPUInputs() {
        delete[] q;
        delete[] k;
        delete[] q_w;
        delete[] k_w;
        delete[] cos_sin;
        delete[] out_q;
        delete[] out_k;
    }

    void operator()() {
        for (int tid = 0; tid < num_tokens; tid++) {
            for (int hid = 0; hid < num_heads; hid++) {
                double x2_q = 0;
                double x2_k = 0;
                int offset = (tid * num_heads + hid) * head_size;
                for (int h = 0; h < head_size; ++h) {
                    auto q_data = q[offset + h];
                    auto k_data = k[offset + h];
                    x2_q += q_data * q_data;
                    x2_k += k_data * k_data;
                }
                double beta_q = (double)1.0 / std::sqrt(x2_q / head_size + eps);
                double beta_k = (double)1.0 / std::sqrt(x2_k / head_size + eps);
                for (int h = rotary_dim; h < head_size; ++h) {
                    out_q[offset + h] = q[offset + h] * beta_q * q_w[h];
                    out_k[offset + h] = k[offset + h] * beta_k * k_w[h];
                }
                int half_rotary_dim = rotary_dim / 2;
                for (int h = 0; h < half_rotary_dim; ++h) {
                    auto cos = cos_sin[tid * head_size + h];
                    auto sin = cos_sin[tid * head_size + h + half_rotary_dim];
                    if (is_neox_style) {
                        auto q1 = q[offset + h] * beta_q * q_w[h];
                        auto k1 = k[offset + h] * beta_k * k_w[h];
                        auto q2 = q[offset + h + half_rotary_dim] * beta_q * q_w[h + half_rotary_dim];
                        auto k2 = k[offset + h + half_rotary_dim] * beta_k * k_w[h + half_rotary_dim];
                        out_q[offset + h] = q1 * cos - q2 * sin;
                        out_q[offset + h + half_rotary_dim] = q2 * cos + q1 * sin;
                        out_k[offset + h] = k1 * cos - k2 * sin;
                        out_k[offset + h + half_rotary_dim] = k2 * cos + k1 * sin;
                    } else {
                        auto q1 = q[offset + h * 2 + 0] * beta_q * q_w[h * 2 + 0];
                        auto k1 = k[offset + h * 2 + 0] * beta_k * k_w[h * 2 + 0];
                        auto q2 = q[offset + h * 2 + 1] * beta_q * q_w[h * 2 + 1];
                        auto k2 = k[offset + h * 2 + 1] * beta_k * k_w[h * 2 + 1];
                        out_q[offset + h * 2 + 0] = q1 * cos - q2 * sin;
                        out_q[offset + h * 2 + 1] = q2 * cos + q1 * sin;
                        out_k[offset + h * 2 + 0] = k1 * cos - k2 * sin;
                        out_k[offset + h * 2 + 1] = k2 * cos + k1 * sin;
                    }
                }
            }
        }
    }
};

template <typename T>
class GPUInputs {
public:
    int num_tokens;
    int num_heads;
    int head_size;
    int rotary_dim;
    bool is_neox_style;
    float eps;
    int numel;
    T *q;
    T *k;
    T *q_w;
    T *k_w;
    T *cos_sin;
    T *out_q;
    T *out_k;

    GPUInputs(int num_tokens, int num_heads, int head_size, int rotary_dim, bool is_neox_style, float eps) :
        num_tokens(num_tokens), num_heads(num_heads), head_size(head_size),
        rotary_dim(rotary_dim), is_neox_style(is_neox_style), eps(eps) {
        numel = num_tokens * num_heads * head_size;
    }

    void allocate() {
        gpuMalloc(&q, numel * sizeof(T));
        gpuMalloc(&k, numel * sizeof(T));
        gpuMalloc(&q_w, head_size * sizeof(T));
        gpuMalloc(&k_w, head_size * sizeof(T));
        gpuMalloc(&cos_sin, num_tokens * head_size * sizeof(T));
        gpuMalloc(&out_q, numel * sizeof(T));
        gpuMalloc(&out_k, numel * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset(CPUInputs<T> &inputs) {
        gpuMemcpy(q, inputs.q, numel * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(k, inputs.k, numel * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(q_w, inputs.q_w, head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(k_w, inputs.k_w, head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(cos_sin, inputs.cos_sin, num_tokens * head_size * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuFree(q);
        gpuFree(k);
        gpuFree(q_w);
        gpuFree(k_w);
        gpuFree(cos_sin);
        gpuFree(out_q);
        gpuFree(out_k);
        gpuDeviceSynchronize();
    }

    float operator()() {
        gpuEvent_t start, stop;
        gpuEventCreate(&start);
        gpuEventCreate(&stop);
        gpuEventRecord(start);

        assert(head_size == rotary_dim);
        rope_rms::fused_rope_rms(q, k, q_w, k_w, cos_sin, out_q, out_k, num_tokens, num_heads, head_size, is_neox_style, eps, 0);
        gpuDeviceSynchronize();

        gpuEventRecord(stop);
        gpuEventSynchronize(stop);
        float ms = 0;
        gpuEventElapsedTime(&ms, start, stop);
        float input_bytes = (numel + head_size) * 2 * sizeof(T) + num_tokens * head_size * sizeof(T);
        float output_bytes = numel * 2 * sizeof(T);
        float gbps = (input_bytes + output_bytes) / 1000.0 / 1000.0 / ms;
        return gbps;
    }

    bool is_error(T out, T ref, float atol) {
        return std::isnan(out) || std::abs(out - ref) > atol;
    }

    bool validate(CPUInputs<T> &inputs, float atol) {
        auto out_q_cpu = new T[numel];
        auto out_k_cpu = new T[numel];
        gpuMemcpy(out_q_cpu, out_q, numel * sizeof(T), gpuMemcpyDeviceToHost);
        gpuMemcpy(out_k_cpu, out_k, numel * sizeof(T), gpuMemcpyDeviceToHost);
        bool val = true;
        for (int i = 0; i < numel; ++i) {
            if (is_error(out_q_cpu[i], inputs.out_q[i], atol)) {
                val = false;
                std::cout << "\n>>> out_q:" << out_q_cpu[i] << ", out_q_ref:" << inputs.out_q[i] << "\n";
                break;
            }
            if (is_error(out_k_cpu[i], inputs.out_k[i], atol)) {
                val = false;
                std::cout << "\n>>> out_k:" << out_k_cpu[i] << ", out_k_ref:" << inputs.out_k[i] << "\n";
                break;
            }
        }
        delete[] out_q_cpu;
        delete[] out_k_cpu;
        return val;
    }
};

template <typename T>
std::tuple<bool, float> runbench(
    int num_tokens,
    int num_heads,
    int head_size,
    int rotary_dim,
    bool is_neox_style,
    float eps,
    float atol = 0.0001) {
    CPUInputs<T> cpu_inputs(num_tokens, num_heads, head_size, rotary_dim, is_neox_style, eps);
    GPUInputs<T> gpu_inputs(num_tokens, num_heads, head_size, rotary_dim, is_neox_style, eps);
    cpu_inputs.allocate();
    gpu_inputs.allocate();
    cpu_inputs.reset();
    gpu_inputs.reset(cpu_inputs);
    cpu_inputs();
    float gbps = gpu_inputs();
    bool val = gpu_inputs.validate(cpu_inputs, atol);
    return {val, gbps};
}

} // namespace test

int main() {
    std::vector<bool> is_neox_styles = {true, false};
    std::vector<int> num_tokens = {513, 1257, 127, 778, 10024, 3};
    std::vector<int> num_heads = {32, 64};
    std::vector<int> head_sizes = {128, 256};
    float eps = 1e-6;
    for (auto is_neox_style : is_neox_styles) {
        for (auto num_token : num_tokens) {
            for (auto num_head : num_heads) {
                for (auto head_size : head_sizes) {
                    std::cout << "num_token:" << num_token << ", num_head:" << num_head << ", head_size:" << head_size << ", is_neox_style:" << is_neox_style;
                    auto [val, gbps] = test::runbench<float>(num_token, num_head, head_size, head_size, is_neox_style, eps);
                    std::cout << ", val:" << val << ", gbps:" << gbps << "\n";
                }
            }
        }
    }
}
