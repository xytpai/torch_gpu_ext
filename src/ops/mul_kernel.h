#pragma once

#include <torch/extension.h>

using namespace at;

void gpu_mul(Tensor &a, Tensor &b, Tensor &out);
