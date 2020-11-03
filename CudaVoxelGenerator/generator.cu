#include "generator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "log.h"
#include "chunk.h"
#include "simplex.cuh"

#define IDX(x,y,z) ((z*(16*16))+(y*16)+x)

static uint8_t* gpuChunkBlocks;

__global__ void kernel_generator_fillChunk_dim3(uint8_t* chunkData, int3 worldPosition)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x; 
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    int worldX = idx + worldPosition.x;
    int worldY = idy + worldPosition.y;
    int worldZ = idz + worldPosition.z;

    float simplexValue = repeaterSimplex(make_float3(static_cast<float>(worldX) / 64.0f, 0.0f, static_cast<float>(worldZ) / 64.0f), 1.0f, 1234, 3, 2.0f, 0.5f);
    simplexValue = ((simplexValue + 1.0f) / 2.0f);

    int height = (int)floorf(simplexValue * 24);

    uint8_t block = 0;
    if (worldY == height) block = 3;
    else if (worldY < height && worldY > height - 3) block = 2;
    else if (worldY < height) block = 1;
    chunkData[IDX(idx, idy, idz)] = block;
}

void cuda_generate_init()
{
    CUDA_CHECK(cudaSetDevice(0));

    CUDA_CHECK(cudaMalloc((void**)&gpuChunkBlocks, CHUNK_BLOCKS));
}

void cuda_generate_clean()
{
    CUDA_CHECK(cudaFree(gpuChunkBlocks));
}

void cuda_generate_chunk(Chunk* chunk)
{
    CUDA_CHECK(cudaMemset(gpuChunkBlocks, 0, CHUNK_BLOCKS));

    int3 chunkPos = chunk->getChunkPosition();
    vec3 worldPosition = vec3(chunkPos.x, chunkPos.y, chunkPos.z) * (float)CHUNK_SIZE;
    
    dim3 block(4, 4, 4);
    dim3 grid(4, 4, 4);

    kernel_generator_fillChunk_dim3<<<grid, block>>>(gpuChunkBlocks, make_int3((int)worldPosition.x, (int)worldPosition.y, (int)worldPosition.z));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(chunk->blocks, gpuChunkBlocks, CHUNK_BLOCKS, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}
