#ifndef H_FILTERS
#define H_FILTERS

#define NUM_SP_FILTERS 4
#define FSP_ID3 0
#define FSP_EDGE3 1
#define FSP_SHARP3 2
#define FSP_LAPGAUS9 3
#define FDP_BLUR5 4

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

#endif // H_FILTERS
