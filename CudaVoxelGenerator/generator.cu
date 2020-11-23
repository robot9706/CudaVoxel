#include "generator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>

#include "log.h"
#include "chunk.h"
#include "simplex.cuh"

static uint8_t* gpuChunkBlocks;

#define CONFIG_SEED_BASE 1234

#define CONFIG_TERRAIN_HEIGHT 24
#define CONFIG_NUM_TREES 2
#define CONFIG_DIRT_HEIGHT 5
#define CONFIG_CAVE_THRESHOLD 0.675f

__constant__ uint8_t terrainConst[CONFIG_TERRAIN_HEIGHT];

__device__ int generate_terrain_height(int worldX, int worldZ)
{
    float simplexValue = repeaterSimplex(make_float3(static_cast<float>(worldX) / 64.0f, 0.0f, static_cast<float>(worldZ) / 64.0f), 1.0f, CONFIG_SEED_BASE, 3, 3.0f, 0.25f);
    simplexValue = ((simplexValue + 1.0f) / 2.0f);

    return (int)floorf(simplexValue * CONFIG_TERRAIN_HEIGHT);
}

__device__ uint32_t hashInt3(int x, int y, int z)
{
    return (x * 607495) + (y * 359609) + (z * 654846);
}

__global__ void kernel_generator_fillChunk_nonOpt_dim3(uint8_t* chunkData, int3 worldPosition)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    int worldX = idx + worldPosition.x;
    int worldY = idy + worldPosition.y;
    int worldZ = idz + worldPosition.z;

    int height = generate_terrain_height(worldX, worldZ);

    uint8_t block = 0;
    if (worldY == height) block = 3;
    else if (worldY < height && worldY > height - 3) block = 2;
    else if (worldY < height) block = 1;
    chunkData[CHUNK_OFFSET(idx, idy, idz)] = block;
}

__global__ void kernel_generator_fillChunk_dim3(uint8_t* chunkData, int3 worldPosition)
{
    __shared__ int terrainHeight[CHUNK_SIZE * CHUNK_SIZE];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;

    terrainHeight[idx + CHUNK_SIZE * idz] = 0;
    __syncthreads();

    int worldX = idx + worldPosition.x;
    int worldY = idy + worldPosition.y;
    int worldZ = idz + worldPosition.z;

    int height = 0;
    if (terrainHeight[idx + CHUNK_SIZE * idz] == 0)
    {
        terrainHeight[idx + CHUNK_SIZE * idz] = height = generate_terrain_height(worldX, worldZ);
    }
    else
    {
        height = terrainHeight[idx + CHUNK_SIZE * idz];
    }

    if (worldY > height)
    {
        return;
    }

    float cave = repeaterSimplex(make_float3(static_cast<float>(worldX) / 64.0f, static_cast<float>(worldY) / 64.0f, static_cast<float>(worldZ) / 64.0f), 1.0f, CONFIG_SEED_BASE, 3, 3.0f, 0.25f);
    cave = ((cave + 1.0f) / 2.0f);
    if (cave >= CONFIG_CAVE_THRESHOLD)
    {
        return;
    }

    int offset = height - worldY;
    chunkData[CHUNK_OFFSET(idx, idy, idz)] = terrainConst[offset];
}

#define O BLOCK_LOG
#define L BLOCK_LEAVES
#define TREE_TEMPLATE_SIZE 5, 7, 5
uint8_t* gpuTreeTemplate;
uint8_t treeTemplate[] = {
    // 0
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, O, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    // 1
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, O, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    // 2
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, O, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,

    // 3
    L, L, L, L, L,
    L, L, L, L, L,
    L, L, O, L, L,
    L, L, L, L, L,
    L, L, L, L, L,

    // 4
    L, L, L, L, L,
    L, L, L, L, L,
    L, L, O, L, L,
    L, L, L, L, L,
    L, L, L, L, L,

    // 5
    0, 0, 0, 0, 0,
    0, 0, L, 0, 0,
    0, L, O, L, 0,
    0, 0, L, 0, 0,
    0, 0, 0, 0, 0,

    // 6
    0, 0, 0, 0, 0,
    0, 0, L, 0, 0,
    0, L, L, L, 0,
    0, 0, L, 0, 0,
    0, 0, 0, 0, 0,
};

__global__ void kernel_decorator_trees(uint8_t* chunkData, int3 worldPosition, uint8_t* templateData, int3 templateSize)
{
    curandState randBase;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t seed = hashInt3(worldPosition.x, CONFIG_SEED_BASE, worldPosition.z);
    curand_init(seed, idx, 0, &randBase);

    int testX = clamp((int)floorf(curand_uniform(&randBase) * (CHUNK_SIZE - 4)) + 2, 0, CHUNK_SIZE - 1);
    int testZ = clamp((int)floorf(curand_uniform(&randBase) * (CHUNK_SIZE - 4)) + 2, 0, CHUNK_SIZE - 1);

    int height = generate_terrain_height(testX + worldPosition.x, testZ + worldPosition.z);
    int treeBottom = height + 1;
    int treeTop = treeBottom + templateSize.y;

    if ((treeTop < worldPosition.y && treeBottom < worldPosition.y) || (treeTop >= worldPosition.y + CHUNK_SIZE && treeBottom >= worldPosition.y + CHUNK_SIZE))
    {
        return;
    }

    int treeX = testX - 2;
    int treeZ = testZ - 2;
    for (int templateY = 0; templateY < templateSize.y; templateY++)
    {
        int blockY = treeBottom + templateY - worldPosition.y;
        if (blockY < 0)
            continue;
        if (blockY >= CHUNK_SIZE)
            break;

        for (int templateX = 0; templateX < templateSize.x; templateX++)
        {
            for (int templateZ = 0; templateZ < templateSize.z; templateZ++)
            {
                uint8_t templateBlock = templateData[templateX + templateZ * templateSize.x + templateY * templateSize.x * templateSize.z];
                if (templateBlock == 0)
                {
                    continue;
                }

                chunkData[CHUNK_OFFSET(treeX + templateX, blockY, treeZ + templateZ)] = templateBlock;
            }
        }
    }
}

void cuda_generate_init()
{
    CUDA_CHECK(cudaSetDevice(0));

    CUDA_CHECK(cudaMalloc((void**)&gpuChunkBlocks, CHUNK_BLOCKS));
    
    CUDA_CHECK(cudaMalloc((void**)&gpuTreeTemplate, sizeof(treeTemplate)));
    CUDA_CHECK(cudaMemcpy(gpuTreeTemplate, treeTemplate, sizeof(treeTemplate), cudaMemcpyKind::cudaMemcpyHostToDevice));

    uint8_t terrainData[CONFIG_TERRAIN_HEIGHT] = { 0 };
    for (int x = 0; x < CONFIG_TERRAIN_HEIGHT; x++)
    {
        if (x == 0)
        {
            terrainData[x] = BLOCK_GRASS;
        }
        else if (x < CONFIG_DIRT_HEIGHT + 1)
        {
            terrainData[x] = BLOCK_DIRT;
        }
        else
        {
            terrainData[x] = BLOCK_STONE;
        }
    }
    cudaMemcpyToSymbol(terrainConst, terrainData, sizeof(uint8_t) * CONFIG_TERRAIN_HEIGHT, 0, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

void cuda_generate_clean()
{
    CUDA_CHECK(cudaFree(gpuChunkBlocks));
    CUDA_CHECK(cudaFree(gpuTreeTemplate));
}

// Elsõ: ~18ms
// Többi átlag: 0.9ms
void cuda_generate_chunk(Chunk* chunk)
{
    CUDA_CHECK(cudaMemset(gpuChunkBlocks, 0, CHUNK_BLOCKS));

    int3 chunkPos = chunk->getChunkPosition();
    vec3 worldPosition = vec3(chunkPos.x, chunkPos.y, chunkPos.z) * (float)CHUNK_SIZE;
    
    dim3 block(4, 4, 4);
    dim3 grid(4, 4, 4);

    // Time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventRecord(start, 0));

    int3 worldBlockPos = make_int3((int)worldPosition.x, (int)worldPosition.y, (int)worldPosition.z);
    kernel_generator_fillChunk_dim3 << <grid, block >> > (gpuChunkBlocks, worldBlockPos);
    //kernel_generator_fillChunk_nonOpt_dim3 << <grid, block >> > (gpuChunkBlocks, worldBlockPos);
    if (CONFIG_NUM_TREES > 0) {
        kernel_decorator_trees << <1, CONFIG_NUM_TREES >> > (gpuChunkBlocks, worldBlockPos, gpuTreeTemplate, make_int3(TREE_TEMPLATE_SIZE));
    }

    // Time
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(stop, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Generate time: %f ms\n", elapsed);

    CUDA_CHECK(cudaMemcpy(chunk->blocks, gpuChunkBlocks, CHUNK_BLOCKS, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}