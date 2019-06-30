#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "filters.h"
#include "kernels.h"
#include "png_utils.h"

// One pixel per thread, row major
__global__ void kernel2(my_png *input, my_png *output, filter *f) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < input->width * input->height) {
		int sums[3] = { 0, 0, 0 };
		int offset = f->dim >> 1;
		int row = idx / input->width, col = idx % input->width;

		for (int r = row - offset, i = 0; r <= row + offset; r++, i++) {
			if (r < 0 || r >= input->height) continue;

			for (int c = col - offset, j = 0; c <= col + offset; c++, j++) {
				if (c < 0 || c >= input->width) continue;

				png_bytep curr_pix = input->pixels[r * input->width + c * 4];
				for (int a = 0; a < 3; a++)
					sums[a] += curr_pix[a] * f->matrix[i * f->dim + j];
			}
		}

		png_bytep pix = output->pixels[row * input->width + col * 4];
		for (int a = 0; a < 3; a++)
			pix[a] = sums[a];
	}
}

__global__ void kernel2n(my_png *output, int *minp, int *maxp) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < output->width * output->height) {
		int row = idx / output->width, col = idx % output->width;
		png_bytep pix = output->pixels[row * output->width + col * 4];

		for (int a = 0; a < 3; a++) {
			if (minp[a] != maxp[a])
				pix[a] = ((pix[a] - minp[a]) * 255) / (maxp[a] - minp[a]);
		}
	}
}
