// !!! This is a file automatically generated by hipify!!!
#include "caffe2/core/hip/context_gpu.h"
#include "caffe2/operators/prepend_dim_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(PrependDim, PrependDimOp<HIPContext>);
REGISTER_HIP_OPERATOR(MergeDim, MergeDimOp<HIPContext>);

} // namespace caffe2