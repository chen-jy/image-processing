#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "checked_utils.h"

void *Malloc(size_t size) {
	void *malloc_ptr = malloc(size);
	if (!malloc_ptr) {
		perror("malloc");
		exit(1);
	}
	return malloc_ptr;
}

FILE *Fopen(const char *pathname, const char *mode) {
	FILE *file = fopen(pathname, mode);
	if (!file) {
		perror("fopen");
		exit(1);
	}
	return file;
}
