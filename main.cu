#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <png.h>
#include <stdio.h>
#include <string>

#include "checked_utils.h"
#include "clock.h"
#include "getopt.h"

using namespace std;

struct my_png {
	int width, height;
	png_byte color_type;
	png_byte bit_depth;
	png_bytep *pixels;
};

void destroy_png(my_png *png) {
	for (int i = 0; i < png->height; i++)
		free(png->pixels[i]);
	free(png->pixels);
	free(png);
}


struct filter {
	int dim;
	int *matrix;
};

struct filter_dp {
	int dim;
	double *matrix;
};

int _id3_m[] = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
filter _id3 = { 3, _id3_m };

int _edge3_m[] = { -1, -1, -1, -1,  8, -1, -1, -1, -1 };
filter _edge3 = { 3, _edge3_m };

int _sharp3_m[] = { 0, -1,  0, -1,  5, -1, 0, -1,  0 };
filter _sharp3 = { 3, _sharp3_m };

int _lapgaus9_m[] = { 0, 1, 1, 2, 2, 2, 1, 1, 0, 1, 2, 4, 5, 5, 5, 4, 2, 1,
	1, 4, 5, 3, 0, 3, 5, 4, 1, 2, 5, 3, -12, -24, -12, 3, 5, 2,
	2, 5, 0, -24, -40, -24, 0, 5, 2, 2, 5, 3, -12, -24, -12, 3, 5, 2,
	1, 4, 5, 3, 0, 3, 5, 4, 1, 1, 2, 4, 5, 5, 5, 4, 2, 1, 0, 1, 1, 2, 2, 2, 1, 1, 0 };
filter _lapgaus9 = { 9, _lapgaus9_m };

double _blur5_m[] = { 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6,
	4, 16, 24, 16, 4, 1, 4, 6, 4, 1 };
filter_dp dp_filter = { 5, _blur5_m };

filter *filters[4] = { &_id3, &_edge3, &_sharp3, &_lapgaus9 };


// PNG reading and writing adapted from https://gist.github.com/niw/5963798
my_png *read_png(string filename) {
	my_png *myPNG = (my_png*)Malloc(sizeof(my_png));
	FILE *file = Fopen(filename.c_str(), "rb");

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) return NULL;
	png_infop info = png_create_info_struct(png);
	if (!info) return NULL;

	if (setjmp(png_jmpbuf(png)))
		return NULL;

	png_init_io(png, file);
	png_read_info(png, info);

	myPNG->width = png_get_image_width(png, info);
	myPNG->height = png_get_image_height(png, info);
	myPNG->color_type = png_get_color_type(png, info);
	myPNG->bit_depth = png_get_bit_depth(png, info);

	if (myPNG->bit_depth == 16)
		png_set_strip_16(png);
	if (myPNG->color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png);
	if (myPNG->color_type == PNG_COLOR_TYPE_GRAY && myPNG->bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png);
	if (png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);

	if (myPNG->color_type == PNG_COLOR_TYPE_RGB ||
		myPNG->color_type == PNG_COLOR_TYPE_GRAY ||
		myPNG->color_type == PNG_COLOR_TYPE_PALETTE) {
		png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
	}
	if (myPNG->color_type == PNG_COLOR_TYPE_GRAY ||
		myPNG->color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
		png_set_gray_to_rgb(png);
	}

	png_read_update_info(png, info);

	myPNG->pixels = (png_bytep*)Malloc(myPNG->height * sizeof(png_bytep));
	for (int i = 0; i < myPNG->height; i++)
		myPNG->pixels[i] = (png_byte*)Malloc(png_get_rowbytes(png, info));

	png_read_image(png, myPNG->pixels);
	png_destroy_read_struct(&png, &info, NULL);

	fclose(file);
	return myPNG;
}

int write_png(string filename, my_png *myPNG) {
	FILE *file = Fopen(filename.c_str(), "wb");

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png) return 1;
	png_infop info = png_create_info_struct(png);
	if (!info) return 1;

	if (setjmp(png_jmpbuf(png)))
		return 1;

	png_init_io(png, file);
	png_set_IHDR(png, info, myPNG->width, myPNG->height, 8, PNG_COLOR_TYPE_RGBA,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(png, info);

	if (!myPNG->pixels)
		return 1;

	png_write_image(png, myPNG->pixels);
	png_write_end(png, NULL);
	png_destroy_write_struct(&png, &info);

	fclose(file);
	return 0;
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

	my_png *png = read_png(input_filename);
	write_png(cpu_filename, png);
	destroy_png(png);

	return 0;
}
