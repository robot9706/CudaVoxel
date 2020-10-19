#pragma once

#ifndef __LOG__
#define __LOG__

#define ERROR_FORMAT(FORMAT, PARAMS) fprintf(stderr, FORMAT, PARAMS); exit(-1);
#define ERROR_TEXT(TEXT) ERROR_FORMAT("%s\n", TEXT)

#define CUDA_CHECK(STATUS) { cudaError_t status = (STATUS); if (status != cudaSuccess) { \
	ERROR_FORMAT("CUDA error! %s\n", cudaGetErrorName(status)); \
} } \

#endif