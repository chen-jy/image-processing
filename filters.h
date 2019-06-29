#ifndef H_FILTER
#define H_FILTER

#define NUM_SP_FILTERS 4

struct filter {
	int dim;
	int *matrix;
};

struct filter_dp {
	int dim;
	double *matrix;
};

extern filter _id3;
extern filter _edge3;
extern filter _sharp3;
extern filter _lapgaus9;
extern filter *filters[NUM_SP_FILTERS];

extern filter_dp dp_filter;

#endif
