#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif
#include <c10/hip/HIPException.h>

namespace torch { namespace cuda { namespace shared {

void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // By splitting the names of these objects into two literals we prevent the
  // HIP rewrite rules from changing these names when building with HIP.

#ifndef __HIP_PLATFORM_HCC__
  py::enum_<cudaOutputMode_t>(cudart, "cuda" "OutputMode")
      .value("KeyValuePair", hipKeyValuePair)
      .value("CSV", hipCSV);
#endif

  py::enum_<hipError_t>(cudart, "cuda" "Error")
      .value("success", hipSuccess);

  cudart.def("cuda" "GetErrorString", hipGetErrorString);
  cudart.def("cuda" "ProfilerStart", hipProfilerStart);
  cudart.def("cuda" "ProfilerStop", hipProfilerStop);
  cudart.def("cuda" "HostRegister", [](uintptr_t ptr, size_t size, unsigned int flags) -> hipError_t {
    return hipHostRegister((void*)ptr, size, flags);
  });
  cudart.def("cuda" "HostUnregister", [](uintptr_t ptr) -> hipError_t {
    return hipHostUnregister((void*)ptr);
  });
  cudart.def("cuda" "StreamCreate", [](uintptr_t ptr) -> hipError_t {
    return hipStreamCreate((hipStream_t*)ptr);
  });
  cudart.def("cuda" "StreamDestroy", [](uintptr_t ptr) -> hipError_t {
    return hipStreamDestroy((hipStream_t)ptr);
  });
#ifndef __HIP_PLATFORM_HCC__
  cudart.def("cuda" "ProfilerInitialize", hipProfilerInitialize);
#endif
  cudart.def("cuda" "MemGetInfo", [](int device) -> std::pair<size_t, size_t> {
    C10_HIP_CHECK(hipGetDevice(&device));
    size_t device_free;
    size_t device_total;
    hipMemGetInfo(&device_free, &device_total);
    return {device_free, device_total};
  });
}

} // namespace shared
} // namespace cuda
} // namespace torch
