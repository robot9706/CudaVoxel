#include "generator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "log.h"
#include "chunk.h"
#include "simplex.cuh"

#define GENERATOR_THREADS_PER_BLOCK 16

#define GET_POSITION(X,Y,Z,WORLD) X = threadIdx.x + WORLD.x; Y = (blockIdx.x % 16) + WORLD.y; Z = (blockIdx.x / 16) + WORLD.z;

static uint8_t* gpuChunkBlocks;

__global__ void kernel_generator_fillChunk(uint8_t* chunkData, int3 worldPosition)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    chunkData[offset] = 1;
}

__global__ void kernel_generator_fillChunkHalf(uint8_t* chunkData, int3 worldPosition)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ, worldPosition);

    if (blockY > 8) 
    {
        return;
    }

    chunkData[offset] = (blockY == 8 ? 3 : 2);
}

__global__ void kernel_generator_hole(uint8_t* chunkData, int3 worldPosition)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ, worldPosition);

    if (blockX > 4 && blockX < 12 && blockZ > 4 && blockZ < 12)
    {
        return;
    }

    chunkData[offset] = 4;
}

__global__ void kernel_generator_simplex3D(uint8_t* chunkData, int3 worldPosition)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ, worldPosition);

    float simplexValue = simplexNoise(make_float3(blockX / 8.0f, blockY / 8.0f, blockZ / 8.0f), 1.0f, 1234);

    if (simplexValue > 0)
    {
        chunkData[offset] = 1;
    }
}

__global__ void kernel_generator_simplex(uint8_t* chunkData, int3 worldPosition)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    int blockX, blockY, blockZ;
    GET_POSITION(blockX, blockY, blockZ, worldPosition);

    //float simplexValue = simplexNoise(make_float3(static_cast<float>(blockX) / 128.0f, 0.0f, static_cast<float>(blockZ) / 128.0f), 1.0f, 1234);
    float simplexValue = repeaterSimplex(make_float3(static_cast<float>(blockX) / 64.0f, 0.0f, static_cast<float>(blockZ) / 64.0f), 1.0f, 1234, 3, 2.0f, 0.5f);
    simplexValue = ((simplexValue + 1.0f) / 2.0f);

    float height = simplexValue * CHUNK_SIZE;

    chunkData[offset] = (blockY <= height) ? 1 : 0;
}

#define XSIZE 16
#define YSIZE 16
#define ZSIZE 16
#define BLKX 16
#define BLKY 1
#define BLKZ 16

#define IDX(x,y,z) ((z*(XSIZE*YSIZE))+(y*XSIZE)+x)
#define GET_POSITION3D(X,Y,Z,WORLD,TIDX,TIDY,TIDZ) X = TIDX; Y = TIDY; Z = TIDZ;

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

    int height = (int)floorf(simplexValue * CHUNK_SIZE);

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

    vec3 worldPosition = chunk->getChunkPosition() * (float)CHUNK_SIZE;
    
    dim3 block(4, 4, 4);
    dim3 grid(4, 4, 4);

    //kernel_generator_simplex << <CHUNK_BLOCKS/GENERATOR_THREADS_PER_BLOCK, GENERATOR_THREADS_PER_BLOCK >> > (gpuChunkBlocks, make_int3((int)worldPosition.x, (int)worldPosition.y, (int)worldPosition.z));
    kernel_generator_fillChunk_dim3<<<grid, block>>>(gpuChunkBlocks, make_int3((int)worldPosition.x, (int)worldPosition.y, (int)worldPosition.z));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(chunk->blocks, gpuChunkBlocks, CHUNK_BLOCKS, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}
