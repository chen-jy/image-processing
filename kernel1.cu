#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "filters.h"
#include "kernels.h"
#include "png_utils.h"

// One thread block, one thread
__global__ void kernel1(my_png *input, my_png *output, filter *f) {
	int offset = f->dim >> 1;

	for (int y = 0; y < input->height; y++) {
		for (int x = 0; x < input->width; x++) {
			int sums[3] = { 0, 0, 0 };

			for (int r = y - offset, i = 0; r <= y + offset; r++, i++) {
				if (r < 0 || r >= input->height) continue;

				for (int c = x - offset, j = 0; c <= x + offset; c++, j++) {
					if (c < 0 || c >= input->width) continue;

					png_bytep curr_pix = input->pixels[r * input->width + c * 4];
					for (int a = 0; a < 3; a++)
						sums[a] += curr_pix[a] * f->matrix[i * f->dim + j];
				}
			}

			png_bytep pix = output->pixels[y * input->width + x * 4];
			for (int a = 0; a < 3; a++)
				pix[a] = sums[a];
		}
	}
}

__global__ void kernel1n(my_png *output, int *minp, int *maxp) {
	for (int y = 0; y < output->height; y++) {
		for (int x = 0; x < output->width; x++) {
			png_bytep pix = output->pixels[y * output->width + x * 4];

			for (int a = 0; a < 3; a++) {
				if (minp[a] != maxp[a])
					pix[a] = ((pix[a] - minp[a]) * 255) / (maxp[a] - minp[a]);
			}
		}
	}
}
