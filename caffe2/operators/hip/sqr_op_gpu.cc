// !!! This is a file automatically generated by hipify!!!
#include "caffe2/operators/sqr_op.h"

#include "caffe2/core/hip/context_gpu.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Sqr,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SqrFunctor<HIPContext>>);

} // namespace caffe2
