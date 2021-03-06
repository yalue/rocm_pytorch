// !!! This is a file automatically generated by hipify!!!
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THH/generic/THHStorageCopy.cpp"
#else

#include <c10/hip/HIPFunctions.h>

void THCStorage_(copyCPU)(THCState *state, THCStorage *self, struct THStorage *src)
{
  THArgCheck(self->nbytes() == src->nbytes(), 2, "size does not match");
  hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  at::cuda::memcpy_and_sync(THCStorage_(data)(state, self),
      THStorage_(data)(src),
      self->nbytes(),
      hipMemcpyHostToDevice,
      stream);

}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                                 \
  void THCStorage_(copy##TYPEC)(                                              \
      THCState * state, THCStorage * self, struct TH##TYPEC##Storage * src) { \
    THCTensor* selfTensor = THCTensor_(newWithStorage1d)(                     \
        state, self, 0, src->nbytes() / sizeof(scalar_t), 1);                 \
    struct TH##TYPEC##Tensor* srcTensor = TH##TYPEC##Tensor_newWithStorage1d( \
        src, 0, src->nbytes() / sizeof(scalar_t), 1);                         \
    THCTensor_(copy)(state, selfTensor, srcTensor);                           \
    TH##TYPEC##Tensor_free(srcTensor);                                        \
    THCTensor_(free)(state, selfTensor);                                      \
  }

// TODO: Add cross-dtype storage copy for complex storage
#if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Float)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Half)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(Bool)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(BFloat16)
#else
  TH_CUDA_STORAGE_IMPLEMENT_COPY(ComplexFloat)
  TH_CUDA_STORAGE_IMPLEMENT_COPY(ComplexDouble)
#endif

void THStorage_(copyCuda)(THCState *state, THStorage *self, struct THCStorage *src)
{
  THArgCheck(self->nbytes() == src->nbytes(), 2, "size does not match");
  hipStream_t stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  at::cuda::memcpy_and_sync(
      THStorage_(data)(self),
      THCStorage_(data)(state, src),
      self->nbytes(),
      hipMemcpyDeviceToHost,
      stream);
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                               \
  void TH_CONCAT_4(TH, TYPEC, Storage_copyCuda, Real)(                        \
      THCState * state, TH##TYPEC##Storage * self, struct THCStorage * src) { \
    TH##TYPEC##Tensor* selfTensor = TH##TYPEC##Tensor_newWithStorage1d(       \
        self, 0, self->nbytes() / sizeof(scalar_t), 1);                       \
    struct THCTensor* srcTensor = THCTensor_(newWithStorage1d)(               \
        state, src, 0, src->nbytes() / sizeof(scalar_t), 1);                  \
    THCTensor_(copy)(state, selfTensor, srcTensor);                           \
    THCTensor_(free)(state, srcTensor);                                       \
    TH##TYPEC##Tensor_free(selfTensor);                                       \
  }

// TODO: Add cross-dtype storage copy for complex storage
#if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Float)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Half)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Bool)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(BFloat16)
#else
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(ComplexFloat)
  TH_CUDA_STORAGE_IMPLEMENT_COPYTO(ComplexDouble)
#endif

#undef TH_CUDA_STORAGE_IMPLEMENT_COPY
#undef TH_CUDA_STORAGE_IMPLEMENT_COPYTO

#endif
