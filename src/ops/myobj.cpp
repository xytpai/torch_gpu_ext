#include "myobj.h"
#include "gpu_loops_kernel_impl.h"
#include "rope_rms_impl.h"

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

int64_t MyObject::value() {
    return value_;
}

void fused_rope_rms(Tensor &qkv, Tensor &qw, Tensor &kw, Tensor &cos_sin, Tensor &positions,
                    int64_t num_tokens, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_size,
                    bool is_neox_style, double eps) {
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(qkv));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        qkv.scalar_type(),
        "fused_rope_rms", [&] {
            rope_rms::fused_rope_rms<scalar_t>(
                qkv.data_ptr<scalar_t>(),
                qw.data_ptr<scalar_t>(),
                kw.data_ptr<scalar_t>(),
                cos_sin.data_ptr<scalar_t>(),
                positions.data_ptr<int64_t>(),
                num_tokens,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_size,
                is_neox_style,
                eps,
                stream);
        });
}
