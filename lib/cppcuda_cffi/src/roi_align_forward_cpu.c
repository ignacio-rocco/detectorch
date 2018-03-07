// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_op.cc
// (Ignacio Rocco)

#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>
#include "cpp/roi_align_cpu_loop.h"

#define real float

int roi_align_forward_cpu(
  THFloatTensor *input,
  THFloatTensor *bottom_rois,
  THFloatTensor *output,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio)
{

  int proposals = THFloatTensor_size(bottom_rois, 0);
  int roi_cols = THFloatTensor_size(bottom_rois, 1);
  int channels = THFloatTensor_size(input, 1);
  int height = THFloatTensor_size(input, 2);
  int width = THFloatTensor_size(input, 3);


  int64_t total_threads = proposals*channels*pooled_height*pooled_width;
  
  roi_align_forward_loop(
    total_threads, 
    THFloatTensor_data(input), 
    THFloatTensor_data(bottom_rois),
    (float)(spatial_scale), 
    channels,
    height, 
    width, 
    pooled_height, 
    pooled_width, 
    sampling_ratio,
    roi_cols,
    THFloatTensor_data(output));

  return 1;
}




