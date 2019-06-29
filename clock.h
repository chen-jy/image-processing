#ifndef H_CLOCK
#define H_CLOCK

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Clock {
public:
	Clock() {
		cudaEventCreate(&event_start);
		cudaEventCreate(&event_stop);
	}

	void start() {
		cudaEventRecord(event_start);
	}

	double stop() {
		float time;

		cudaEventRecord(event_stop);
		cudaEventSynchronize(event_stop);
		cudaEventElapsedTime(&time, event_start, event_stop);

		return time;
	}

private:
	cudaEvent_t event_start, event_stop;
};

#endif // H_CLOCK
