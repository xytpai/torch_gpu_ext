#pragma once

#include <torch/extension.h>

class MyObject : public torch::CustomClassHolder {
public:
    MyObject(int64_t value) :
        value_(value) {
    }
    int64_t value();

private:
    int64_t value_;
};
