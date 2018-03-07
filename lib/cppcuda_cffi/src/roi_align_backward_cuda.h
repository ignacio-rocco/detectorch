int roi_align_backward_cuda(
  THCudaTensor *bottom_rois,
  THCudaTensor *grad_output, // gradient of the output of the layer
  THCudaTensor *output,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);