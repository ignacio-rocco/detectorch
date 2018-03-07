// Adapted from https://github.com/caffe2/caffe2/blob/master/caffe2/operators/roi_align_op.cu
// (Ignacio Rocco)
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>


// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;
const int CUDA_MAX_BLOCKS = 65535;

inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__host__ __device__ __forceinline__ float myfmin(float a, float b) {
  return a > b ? b : a;
}

__host__ __device__ __forceinline__ float myfmax(float a, float b) {
  return a > b ? a : b;
}

__device__ float bilinear_interpolate(
    const float* bottom_data,
    const int height,
    const int width,
    float y,
    float x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float)x_low;
  } else {
    x_high = x_low + 1;
  }

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  float v1 = bottom_data[y_low * width + x_low];
  float v2 = bottom_data[y_low * width + x_high];
  float v3 = bottom_data[y_high * width + x_low];
  float v4 = bottom_data[y_high * width + x_high];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

__global__ void roi_align_forward_kernel(
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
    float* top_data) // output
 {
//  CUDA_1D_KERNEL_LOOP(index, nthreads) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < outputElements;
       index += blockDim.x * gridDim.x)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const float* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_bottom_rois[1] * spatial_scale;
    float roi_start_h = offset_bottom_rois[2] * spatial_scale;
    float roi_end_w = offset_bottom_rois[3] * spatial_scale;
    float roi_end_h = offset_bottom_rois[4] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    float roi_width = myfmax(roi_end_w - roi_start_w, (float)1.);
    float roi_height = myfmax(roi_end_h - roi_start_h, (float)1.);
    float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    const float* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceilf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceilf(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    float output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const float y = roi_start_h + ph * bin_size_h +
          static_cast<float>(iy + .5f) * bin_size_h /
              static_cast<float>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
            static_cast<float>(ix + .5f) * bin_size_w /
                static_cast<float>(roi_bin_grid_w);

        float val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

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
    cudaStream_t stream)
{

  int64_t blocks = myfmin(GET_BLOCKS(outputElements),CUDA_MAX_BLOCKS);
  
  roi_align_forward_kernel<<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
    outputElements,
    bottom_data, // input tensor
    bottom_rois, // input rois
    spatial_scale,
    channels,
    height,
    width,
    pooled_height,
    pooled_width,
    sampling_ratio,
    top_data);
  
  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;

}


#ifdef __cplusplus
}
#endif