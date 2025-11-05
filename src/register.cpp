#include <Python.h>
#include "add_kernel.h"
#include "mul_kernel.h"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject *PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,   /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}
}

TORCH_LIBRARY(torch_gpu_ext, m) {
    m.def("gpu_add(Tensor a, Tensor b, Tensor out) -> ()");
    m.def("gpu_mul(Tensor a, Tensor b, Tensor out) -> ()");
}

TORCH_LIBRARY_IMPL(torch_gpu_ext, CUDA, m) {
    m.impl("gpu_add", &gpu_add);
    m.impl("gpu_mul", &gpu_mul);
}
