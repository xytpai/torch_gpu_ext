#pragma once

#include <torch/extension.h>

using namespace at;

void gpu_add(Tensor &a, Tensor &b, Tensor &out);
