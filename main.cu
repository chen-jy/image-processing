#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdio.h>
#include <string>

#include "clock.h"
#include "getopt.h"

using namespace std;

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

	return 0;
}
