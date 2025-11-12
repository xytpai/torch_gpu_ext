#pragma once

#include <torch/extension.h>

using namespace at;

class MyObject : public torch::CustomClassHolder {
public:
    MyObject(int64_t value) :
        value_(value) {
    }
    int64_t value();

private:
    int64_t value_;
};

void fused_rope_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                    bool is_neox_style, double eps);
