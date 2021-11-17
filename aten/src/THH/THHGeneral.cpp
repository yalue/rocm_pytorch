// !!! This is a file automatically generated by hipify!!!
#include <THH/THHGeneral.h>
#include <TH/TH.h>
#include <THH/THHAllocator.h>
#include <THH/THHCachingHostAllocator.h>
#include <THH/THHGeneral.hpp>

#include <c10/hip/HIPFunctions.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>

#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <stdlib.h>
#include <stdint.h>

/* Size of scratch space available in global memory per each SM + stream */
#define MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

/* Minimum amount of scratch space per device. Total scratch memory per
 * device is either this amount, or the # of SMs * the space per SM defined
 * above, whichever is greater.*/
#define MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE 32768 * sizeof(float)

/* Maximum number of P2P connections (if there are more than 9 then P2P is
 * enabled in groups of 8). */
#define THC_CUDA_MAX_PEER_SIZE 8

void THCState_free(THCState* state)
{
  free(state);
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

THCState* THCState_alloc(void)
{
  THCState* state = (THCState*) calloc(1, sizeof(THCState));
  return state;
}

void THCudaInit(THCState* state)
{
  if (!state->hipHostAllocator) {
    state->hipHostAllocator = getTHCCachingHostAllocator();
  }

  // We want to throw if there are no GPUs
  int numDevices = static_cast<int>(c10::hip::device_count_ensure_non_zero());
  state->numDevices = numDevices;

  c10::hip::HIPCachingAllocator::init(numDevices);

  int device = 0;
  THCudaCheck(hipGetDevice(&device));

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    calloc(numDevices, sizeof(THCCudaResourcesPerDevice));

  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  state->p2pAccessEnabled = (int**) calloc(numDevices, sizeof(int*));
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) calloc(numDevices, sizeof(int));
    for (int j = 0; j < numDevices; ++j)
      if (i == j)
        state->p2pAccessEnabled[i][j] = 1;
      else
        state->p2pAccessEnabled[i][j] = -1;
  }

  for (int i = 0; i < numDevices; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    THCudaCheck(hipSetDevice(i));

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device. We guarantee a
       minimum of 128kb of space per device, but to future-proof against
       future architectures that may have huge #s of SMs, we guarantee that
       we have at least 16 bytes for each SM. */
    int numSM = at::cuda::getDeviceProperties(i)->multiProcessorCount;
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;
  }

  /* Restore to previous device */
  THCudaCheck(hipSetDevice(device));
}

void THCudaShutdown(THCState* state)
{

  int deviceCount = 0;
  int prevDev = -1;
  THCudaCheck(hipGetDevice(&prevDev));
  THCudaCheck(hipGetDeviceCount(&deviceCount));

  /* cleanup p2p access state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    free(state->p2pAccessEnabled[dev]);
  }
  free(state->p2pAccessEnabled);

  free(state->resourcesPerDevice);
  c10::hip::HIPCachingAllocator::emptyCache();
  if (state->hipHostAllocator == getTHCCachingHostAllocator()) {
    THCCachingHostAllocator_emptyCache();
  }

  THCudaCheck(hipSetDevice(prevDev));
}

int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess)
{
  if (dev < 0 || dev >= state->numDevices) {
    THError("%d is not a device", dev);
  }
  if (devToAccess < 0 || devToAccess >= state->numDevices) {
    THError("%d is not a device", devToAccess);
  }
  if (state->p2pAccessEnabled[dev][devToAccess] == -1) {
    int prevDev = 0;
    THCudaCheck(hipGetDevice(&prevDev));
    THCudaCheck(hipSetDevice(dev));

    int access = 0;
    THCudaCheck(hipDeviceCanAccessPeer(&access, dev, devToAccess));
    if (access) {
      hipError_t err = hipDeviceEnablePeerAccess(devToAccess, 0);
      if (err == hipErrorPeerAccessAlreadyEnabled) {
        // ignore and clear the error if access was already enabled
        hipGetLastError();
      } else {
        THCudaCheck(err);
      }
      state->p2pAccessEnabled[dev][devToAccess] = 1;
    } else {
      state->p2pAccessEnabled[dev][devToAccess] = 0;
    }

    THCudaCheck(hipSetDevice(prevDev));
  }
  return state->p2pAccessEnabled[dev][devToAccess];
}

c10::Allocator* THCState_getCudaHostAllocator(THCState* state)
{
  return state->hipHostAllocator;
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  return &(state->resourcesPerDevice[device]);
}

size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THCudaCheck(hipGetDevice(&device));
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  return res->scratchSpacePerStream;
}

void __THCudaCheck(hipError_t err, const char *file, const int line)
{
  if(err != hipSuccess)
  {
    static int alreadyFailed = 0;
    if(!alreadyFailed) {
      fprintf(stderr, "THCudaCheck FAIL file=%s line=%i error=%i : %s\n", file, line, err, hipGetErrorString(err));
      alreadyFailed = 1;
    }
    _THError(file, line, "cuda runtime error (%d) : %s", err,
             hipGetErrorString(err));
  }
}

void __THCudaCheckWarn(hipError_t err, const char *file, const int line)
{
  if(err != hipSuccess)
  {
    fprintf(stderr, "THCudaCheckWarn FAIL file=%s line=%i error=%i : %s\n", file, line, err, hipGetErrorString(err));
  }
}

void __THCublasCheck(rocblas_status status, const char *file, const int line)
{
  if(status != rocblas_status_success)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case rocblas_status_invalid_handle:
        errmsg = "library not initialized";
        break;

      case rocblas_status_memory_error:
        errmsg = "resource allocation failed";
        break;

      case rocblas_status_invalid_pointer:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case rocblas_status_not_implemented:
        errmsg = "an absent device architectural feature is required";
        break;

#ifndef __HIP_PLATFORM_HCC__
      case rocblas_status_internal_error:
        errmsg = "an access to GPU memory space failed";
        break;

      case rocblas_status_internal_error:
        errmsg = "the GPU program failed to execute";
        break;
#endif

      case rocblas_status_internal_error:
        errmsg = "an internal operation failed";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cublas runtime error : %s", errmsg);
  }
}

void __THCusparseCheck(hipsparseStatus_t status, const char *file, const int line)
{
  if(status != HIPSPARSE_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case HIPSPARSE_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case HIPSPARSE_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case HIPSPARSE_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case HIPSPARSE_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case HIPSPARSE_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case HIPSPARSE_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case HIPSPARSE_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        errmsg = "the matrix type is not supported by this function";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cusparse runtime error : %s", errmsg);
  }
}

void* THCudaMalloc(THCState *state, size_t size)
{
  return c10::hip::HIPCachingAllocator::raw_alloc(size);
}

void THCudaFree(THCState *state, void* ptr) {
  c10::hip::HIPCachingAllocator::raw_delete(ptr);
}

at::DataPtr THCudaHostAlloc(THCState *state, size_t size)
{
  THCudaCheck(hipGetLastError());
  c10::Allocator* allocator = state->hipHostAllocator;
  return allocator->allocate(size);
}

void THCudaHostRecord(THCState *state, void *ptr) {
  if (state->hipHostAllocator == getTHCCachingHostAllocator()) {
    THCCachingHostAllocator_recordEvent(ptr, at::hip::getCurrentHIPStreamMasqueradingAsCUDA());
  }
}

#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE

#include <THH/THHStorage.cpp>
#include <THH/THHAllocator.cpp>