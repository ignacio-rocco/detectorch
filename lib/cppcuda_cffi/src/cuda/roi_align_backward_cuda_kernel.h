// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_gradient_op.cu
// (Ignacio Rocco)
#ifdef __cplusplus
extern "C" {
#endif

int launch_roi_align_backward_cuda(
    const int nthreads,
    const float* top_diff,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    float* bottom_diff,
    const float* bottom_rois,
    int roi_cols,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif