
int roi_align_forward_cpu(
  THFloatTensor *input,
  THFloatTensor *bottom_rois,
  THFloatTensor *output,
  int64_t pooled_height,
  int64_t pooled_width,
  double spatial_scale,
  int64_t sampling_ratio);