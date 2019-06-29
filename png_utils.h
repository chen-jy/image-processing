#ifndef H_PNG_UTILS
#define H_PNG_UTILS

#include <png.h>
#include <string>

using namespace std;

struct my_png {
	int width, height;
	png_byte color_type;
	png_byte bit_depth;
	png_bytep *pixels;
};

void destroy_png(my_png *png);
my_png *read_png(string filename);
int write_png(string filename, my_png *myPNG);

#endif // H_PNG_UTILS
