// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_op.cu
// (Ignacio Rocco)
#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "cuda/roi_align_forward_cuda_kernel.h"


#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;


int roi_align_forward_cuda(
  THCudaTensor *input,
  THCudaTensor *bottom_rois,
  THCudaTensor *output,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio)
{

  int proposals = THCudaTensor_size(state, bottom_rois, 0);
  int channels = THCudaTensor_size(state, input, 1);
  int height = THCudaTensor_size(state, input, 2);
  int width = THCudaTensor_size(state, input, 3);

  
  int64_t total_threads = proposals*channels*pooled_height*pooled_width;
  
  cudaStream_t stream = THCState_getCurrentStream(state);

  launch_roi_align_forward_cuda(
    total_threads, 
    THCudaTensor_data(state, input), 
    THCudaTensor_data(state, bottom_rois),
    (float)(spatial_scale), 
    channels,
    height, 
    width, 
    pooled_height, 
    pooled_width, 
    sampling_ratio,
    THCudaTensor_data(state, output),
    stream);

  return 1;
}

