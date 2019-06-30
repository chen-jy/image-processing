#ifndef H_KERNELS
#define H_KERNELS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "filters.h"
#include "png_utils.h"

__global__ void kernel1(my_png *input, my_png *output, filter *f);
__global__ void kernel1n(my_png *output, int *minp, int *maxp);

__global__ void kernel2(my_png *input, my_png *output, filter *f);
__global__ void kernel2n(my_png *output, int *minp, int *maxp);

#endif // H_KERNELS
