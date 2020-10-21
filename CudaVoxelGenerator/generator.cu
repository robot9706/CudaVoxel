#include "generator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "log.h"
#include "chunk.h"
#include "simplex.cuh"

#define GENERATOR_THREADS_PER_BLOCK 16

#define GET_POSITION(X,Y,Z) X = threadIdx.x; Y = blockIdx.x % 16; Z = blockIdx.x / 16;

static uint8_t* gpuChunkBlocks;

__global__ void kernel_generator_fillChunk(uint8_t* chunkData)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    chunkData[offset] = 1;
}

__global__ void kernel_generator_fillChunkHalf(uint8_t* chunkData)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ);

    if (blockY > 8) 
    {
        return;
    }

    chunkData[offset] = (blockY == 8 ? 3 : 2);
}

__global__ void kernel_generator_hole(uint8_t* chunkData)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ);

    if (blockX > 4 && blockX < 12 && blockZ > 4 && blockZ < 12)
    {
        return;
    }

    chunkData[offset] = 4;
}

__global__ void kernel_generator_simplex(uint8_t* chunkData)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ);

    float simplexValue = simplexNoise(make_float3(blockX / 128.0f, 0.0f, blockZ / 128.0f), 1.0f, 1234);
    simplexValue = ((simplexValue + 1.0f) / 2.0f);

    float height = simplexValue * CHUNK_SIZE;

    if (blockY <= height)
    {
        chunkData[offset] = 1;
    }
}

__global__ void kernel_generator_simplex3D(uint8_t* chunkData)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ);

    float simplexValue = simplexNoise(make_float3(blockX / 8.0f, blockY / 8.0f, blockZ / 8.0f), 1.0f, 1234);

    if (simplexValue > 0)
    {
        chunkData[offset] = 1;
    }
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
    
    kernel_generator_simplex3D << <CHUNK_BLOCKS/GENERATOR_THREADS_PER_BLOCK, GENERATOR_THREADS_PER_BLOCK >> > (gpuChunkBlocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(chunk->blocks, gpuChunkBlocks, CHUNK_BLOCKS, cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
