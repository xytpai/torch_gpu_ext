#include <torch/extension.h>
#include <ATen/ATen.h>
#include "add_kernel.h"

TORCH_LIBRARY(pytorch_gpu_ext, m) {
    m.def("gpu_add(Tensor a, Tensor b, Tensor out) -> ()");
}

TORCH_LIBRARY_IMPL(pytorch_gpu_ext, CUDA, m) {
    m.impl("gpu_add", &gpu_add);
}

PYBIND11_MODULE(pytorch_gpu_ext, m) {
}
