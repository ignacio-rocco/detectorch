int roi_align_forward_cuda(
  THCudaTensor *input,
  THCudaTensor *bottom_rois,
  THCudaTensor *output,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);