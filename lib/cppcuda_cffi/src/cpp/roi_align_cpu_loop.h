#ifdef __cplusplus
extern "C" {
#endif
	
void roi_align_forward_loop(
    const int outputElements,
    const float* bottom_data, // input tensor
    const float* bottom_rois,  // input rois
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const int roi_cols, // rois can have 4 or 5 columns
    float* top_data);

#ifdef __cplusplus
}
#endif

