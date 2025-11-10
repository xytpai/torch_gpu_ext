#include "rope_rms_impl.h"

namespace test {

template <typename T>
class CPUInputs {
public:
    int seq_len;
    int num_heads;
    int head_dim;
    int numel;
    float eps;
    T *q;
    T *k;
    T *q_w;
    T *k_w;
    T *out_q;
    T *out_k;

    CPUInputs(int seq_len, int num_heads, int head_dim, float eps) :
        seq_len(seq_len), num_heads(num_heads), head_dim(head_dim), eps(eps) {
        numel = seq_len * num_heads * head_dim;
    }

    void allocate() {
        q = new T[numel];
        k = new T[numel];
        q_w = new T[head_dim];
        k_w = new T[head_dim];
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
        for (int i = 0; i < head_dim; ++i) {
            q_w[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
            k_w[i] = 2.f * ((rand() / (float)INT_MAX) - 0.5f);
        }
    }

    ~CPUInputs() {
        delete[] q;
        delete[] k;
        delete[] q_w;
        delete[] k_w;
        delete[] out_q;
        delete[] out_k;
    }

    void operator()() {
        for (int sid = 0; sid < seq_len; sid++) {
            for (int hid = 0; hid < num_heads; hid++) {
                double x2_q = 0;
                double x2_k = 0;
                int offset = (sid * num_heads + hid) * head_dim;
                for (int h = 0; h < head_dim; ++h) {
                    auto q_data = q[offset + h];
                    auto k_data = k[offset + h];
                    x2_q += q_data * q_data;
                    x2_k += k_data * k_data;
                }
                double beta_q = (double)1.0 / std::sqrt(x2_q / head_dim + eps);
                double beta_k = (double)1.0 / std::sqrt(x2_k / head_dim + eps);
                for (int h = 0; h < head_dim; ++h) {
                    out_q[offset + h] = q[offset + h] * beta_q * q_w[h];
                    out_k[offset + h] = k[offset + h] * beta_k * k_w[h];
                }
            }
        }
    }
};

template <typename T>
class GPUInputs {
public:
    int seq_len;
    int num_heads;
    int head_dim;
    int numel;
    float eps;
    T *q;
    T *k;
    T *q_w;
    T *k_w;
    T *out_q;
    T *out_k;

    GPUInputs(int seq_len, int num_heads, int head_dim, float eps) :
        seq_len(seq_len), num_heads(num_heads), head_dim(head_dim), eps(eps) {
        numel = seq_len * num_heads * head_dim;
    }

    void allocate() {
        gpuMalloc(&q, numel * sizeof(T));
        gpuMalloc(&k, numel * sizeof(T));
        gpuMalloc(&q_w, head_dim * sizeof(T));
        gpuMalloc(&k_w, head_dim * sizeof(T));
        gpuMalloc(&out_q, numel * sizeof(T));
        gpuMalloc(&out_k, numel * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset(CPUInputs<T> &inputs) {
        gpuMemcpy(q, inputs.q, numel * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(k, inputs.k, numel * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(q_w, inputs.q_w, head_dim * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(k_w, inputs.k_w, head_dim * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuFree(q);
        gpuFree(k);
        gpuFree(q_w);
        gpuFree(k_w);
        gpuFree(out_q);
        gpuFree(out_k);
        gpuDeviceSynchronize();
    }

    float operator()() {
        gpuEvent_t start, stop;
        gpuEventCreate(&start);
        gpuEventCreate(&stop);
        gpuEventRecord(start);

        rope_rms::fused_rope_rms(q, k, q_w, k_w, out_q, out_k, seq_len, num_heads, head_dim, eps, 0);
        gpuDeviceSynchronize();

        gpuEventRecord(stop);
        gpuEventSynchronize(stop);
        float ms = 0;
        gpuEventElapsedTime(&ms, start, stop);
        float input_bytes = (numel + head_dim) * 2 * sizeof(T);
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
std::tuple<bool, float> runbench(int seq_len, int num_heads, int head_dim, float eps = 1e-6, float atol = 0.0001) {
    CPUInputs<T> cpu_inputs(seq_len, num_heads, head_dim, eps);
    GPUInputs<T> gpu_inputs(seq_len, num_heads, head_dim, eps);
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
    std::vector<int> seq_lens = {513, 1257, 127, 778, 10024, 3};
    std::vector<int> num_heads = {32, 64};
    std::vector<int> head_dims = {128, 256};
    for (auto seq_len : seq_lens) {
        for (auto num_head : num_heads) {
            for (auto head_dim : head_dims) {
                std::cout << "seq_len:" << seq_len << ", num_head:" << num_head << ", head_dim:" << head_dim;
                auto [val, gbps] = test::runbench<float>(seq_len, num_head, head_dim);
                std::cout << ", val:" << val << ", gbps:" << gbps << "\n";
            }
        }
    }
}
