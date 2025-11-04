#include "add_kernel.h"
#include "add_kernel_impl.h"

using namespace at;

void gpu_add(Tensor &a, Tensor &b, Tensor &out) {
    int64_t numel = a.numel();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        a.scalar_type(),
        "gpu_add", [&] {
            gpu_add<scalar_t>(
                a.const_data_ptr<scalar_t>(),
                b.const_data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                numel);
        });
}
