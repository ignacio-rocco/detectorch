// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_gradient_op.cu
// (Ignacio Rocco)
#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "cuda/roi_align_backward_cuda_kernel.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int roi_align_backward_cuda(
  THCudaTensor *bottom_rois,
  THCudaTensor *grad_output, // gradient of the output of the layer
  THCudaTensor *output,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio)
{

  // ROIs is the set of region proposals to process. It is a 2D Tensor where the first
  // dim is the # of proposals, and the second dim is the proposal itself in the form
  // [batch_index startW startH endW endH]
  int num_rois = THCudaTensor_size(state, bottom_rois, 0);
  int roi_cols = THCudaTensor_size(state, bottom_rois, 1);
  int channels = THCudaTensor_size(state, output, 1);
  int height = THCudaTensor_size(state, output, 2);
  int width = THCudaTensor_size(state, output, 3);

  
  int64_t total_threads = num_rois*channels*pooled_height*pooled_width;

  cudaStream_t stream = THCState_getCurrentStream(state);

  launch_roi_align_backward_cuda(
    total_threads,
    THCudaTensor_data(state, grad_output),
    num_rois,
    spatial_scale,
    channels,
    height,
    width,
    pooled_height,
    pooled_width,
    sampling_ratio,
    THCudaTensor_data(state, output),
    THCudaTensor_data(state, bottom_rois),
    roi_cols,
    stream);

  return 1;
}
