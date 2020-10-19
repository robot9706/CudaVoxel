#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "main.h"

#include <stdint.h>
using namespace std;

#define CHUNK_SIZE 16
#define CHUNK_VOXELS (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

__global__ void generateChunkKernel(uint8_t* chunkData)
{
    int thread = threadIdx.x;

    int start = thread * (CHUNK_SIZE * CHUNK_SIZE);
    for (int x = start; x < start + (CHUNK_SIZE * CHUNK_SIZE); x++) {
        chunkData[x] = (thread+1);
    }
}