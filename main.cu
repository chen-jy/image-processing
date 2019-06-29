#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdio.h>
#include <string>

#include "clock.h"
#include "filters.h"
#include "getopt.h"
#include "png_utils.h"

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

	my_png *png = read_png(input_filename);
	write_png(cpu_filename, png);
	destroy_png(png);

	return 0;
}
