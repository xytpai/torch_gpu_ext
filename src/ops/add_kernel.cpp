#include "gpu_loops_kernel_impl.h"
#include <torch/extension.h>

using namespace at;

template <typename T>
struct add_func {
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

void gpu_add(Tensor &a, Tensor &b, Tensor &out) {
    int64_t numel = a.numel();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16,
        kHalf,
        a.scalar_type(),
        "gpu_add", [&] {
            gpu_loops<scalar_t>(
                a.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                (size_t)numel,
                add_func<scalar_t>());
        });
}
