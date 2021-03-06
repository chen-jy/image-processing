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

void print_runtime(int kernel, int cpu_time_ser, int cpu_time_par, float transfer_in,
	float gpu_time, float transfer_out) {
	printf("KERNEL %d =========================\n", kernel);
	printf("Baseline:       %f\n", cpu_time_ser / NUM_ITERATIONS);
	printf("CPU time:       %f\n", cpu_time_par / NUM_ITERATIONS);
	printf("GPU time:       %f\n", gpu_time / NUM_ITERATIONS);
	printf("Transfer in:    %f\n", transfer_in / NUM_ITERATIONS);
	printf("Transfer out:   %f\n", transfer_out / NUM_ITERATIONS);
	printf("GPU speedup:    %f\n", cpu_time_par / gpu_time);
	printf("Total speedup:  %f\n\n", cpu_time_par /
		(transfer_in + gpu_time + transfer_out));
}

void run_convolution(int kernel, my_png *input, my_png *output, filter *filter) {
	int threads, blocks;
	switch (kernel) {
	case 1:
		threads = blocks = 1;
		break;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	switch (kernel) {
	case 1:
		kernel1<<<dimGrid, dimBlock>>>(input, output, filter);
		break;
	}
}

void run_normalization(int kernel, my_png *output, int *minp, int *maxp) {
	int threads, blocks;
	switch (kernel) {
	case 1:
		threads = blocks = 1;
		break;
	}

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	switch (kernel) {
	case 1:
		kernel1n<<<dimGrid, dimBlock>>>(output, minp, maxp);
		break;
	}
}

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
	float time_baseline[NUM_KERNELS] = {};
	float time_cpu[NUM_KERNELS] = {}, time_gpu[NUM_KERNELS] = {};
	float time_in[NUM_KERNELS] = {}, time_out[NUM_KERNELS] = {};

	my_png *input_png = read_png(input_filename);
	if (!input_png) {
		fprintf(stderr, "Error: failed to read PNG\n");
		exit(2);
	}

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
	int *minp, *maxp;

	cudaMalloc(&gpu_input, sizeof(my_png));
	cudaMalloc(&(gpu_input->pixels), image_size);
	cudaMalloc(&gpu_output, sizeof(my_png));
	cudaMalloc(&(gpu_output->pixels), image_size);
	cudaMalloc(&gpu_filter, sizeof(filter));
	cudaMalloc(&(gpu_filter->matrix), filter_size);
	cudaMalloc(&minp, 3 * sizeof(int));
	cudaMalloc(&maxp, 3 * sizeof(int));

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

			clock.start();
			run_convolution(kernel + 1, gpu_input, gpu_output, gpu_filter);
			time_gpu[kernel] += clock.stop();

			// Find min and max here

			clock.start();
			run_normalization(kernel + 1, gpu_output, minp, maxp);
			time_gpu[kernel] += clock.stop();

			clock.start();
			cudaMemcpy(output_png->pixels, gpu_output->pixels, image_size,
				cudaMemcpyDeviceToHost);
			time_out[kernel] += clock.stop();
		}

		if (write_png(gpu_filename + "-kernel" + to_string(kernel + 1) + ".png",
			output_png)) {
			fprintf(stderr, "Error: failed to write PNG\n");
			exit(3);
		}
	}

	for (int kernel = 0; kernel < NUM_KERNELS; kernel++) {
		print_runtime(kernel + 1, time_baseline[kernel], time_cpu[kernel],
			time_in[kernel], time_gpu[kernel], time_out[kernel]);
	}

	cudaFree(gpu_input->pixels);
	cudaFree(gpu_input);
	cudaFree(gpu_output->pixels);
	cudaFree(gpu_output);
	cudaFree(gpu_filter->matrix);
	cudaFree(gpu_filter);
	cudaFree(minp);
	cudaFree(maxp);

	destroy_png(input_png);
	destroy_png(output_png);

	return 0;
}
