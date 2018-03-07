// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_op.cu
// (Ignacio Rocco)
#ifdef __cplusplus
extern "C" {
#endif

int launch_roi_align_forward_cuda(
    const int outputElements,
    const float* bottom_data, // input tensor
    const float* bottom_rois, // input rois
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    float* top_data,
    cudaStream_t stream);


#ifdef __cplusplus
}
#endif