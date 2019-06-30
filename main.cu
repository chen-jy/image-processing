#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdio.h>
#include <string>

#include "checked_utils.h"
#include "clock.h"
#include "filters.h"
#include "getopt.h"
#include "kernels.h"
#include "png_utils.h"

#define NUM_KERNELS 1
#define NUM_ITERATIONS 5
#define MAX_BLOCK_SIZE 65535
#define MAX_THREADS 512

#define FILTER FSP_ID3

using namespace std;

void init() {
	for (int i = 0; i < 25; i++)
		dp_filter.matrix[i] *= (1.0 / 256.0);
}

int main(int argc, char *argv[]) {
	string input_filename, cpu_filename, gpu_filename;
	if (argc < 3) {
		printf("Usage: %s -i <input_file> -o <output_file>\n", argv[0]);
		return 1;
	}

	int c;
	while ((c = getopt(argc, argv, "i:o:")) != -1) {
		switch (c) {
		case 'i':
			input_filename = string(optarg);
			break;
		case 'o':
			cpu_filename = string(optarg);
			gpu_filename = string(optarg);
			break;
		default:
			printf("Usage: %s -i <input_file> -o <output_file>\n", argv[0]);
			return 1;
		}
	}

	init();
	input_filename += ".png";

	Clock clock;
	float time_in[NUM_KERNELS], time_gpu[NUM_KERNELS], time_out[NUM_KERNELS];

	my_png *input_png = read_png(input_filename); // Error check
	size_t image_size = input_png->width * input_png->height * sizeof(png_bytep);
	size_t filter_size = filters[FILTER]->dim * filters[FILTER]->dim * sizeof(int);

	my_png *output_png = (my_png*)Malloc(sizeof(my_png));
	output_png->width = input_png->width;
	output_png->height = input_png->height;
	output_png->color_type = input_png->color_type;
	output_png->bit_depth = input_png->bit_depth;
	output_png->pixels = (png_bytep*)Malloc(image_size);

	my_png *gpu_input, *gpu_output;
	filter *gpu_filter;

	cudaMalloc(&gpu_input, sizeof(my_png));
	cudaMalloc(&(gpu_input->pixels), image_size);
	cudaMalloc(&gpu_output, sizeof(my_png));
	cudaMalloc(&(gpu_output->pixels), image_size);
	cudaMalloc(&gpu_filter, sizeof(filter));
	cudaMalloc(&(gpu_filter->matrix), filter_size);

	for (int kernel = 0; kernel < NUM_KERNELS; kernel++) {
		for (int i = 0; i < NUM_ITERATIONS; i++) {
			clock.start();
			cudaMemcpy(gpu_input, input_png, sizeof(png_bytep),
				cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_input->pixels, input_png->pixels, image_size,
				cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_output, output_png, sizeof(png_bytep),
				cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_output->pixels, output_png->pixels, image_size,
				cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_filter, filters[FILTER], sizeof(filter),
				cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_filter->matrix, filters[FILTER]->matrix, filter_size,
				cudaMemcpyHostToDevice);
			time_in[kernel] += clock.stop();

			// Process the image here
			clock.start();
			kernel1<<<1, 1>>>(gpu_input, gpu_output, gpu_filter);
			// Get the extrema and normalize the image
			time_gpu[kernel] += clock.stop();

			clock.start();
			cudaMemcpy(output_png->pixels, gpu_output->pixels, image_size,
				cudaMemcpyDeviceToHost);
			time_out[kernel] += clock.stop();
		}

		write_png(gpu_filename + "-kernel" + to_string(kernel + 1) + ".png", output_png);
	}

	destroy_png(input_png);
	destroy_png(output_png);

	return 0;
}
