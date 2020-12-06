#include "generator_cpu.h"
#include "generator_config.h"

#include <Windows.h>

// https://github.com/SRombauts/SimplexNoise/blob/master/references/SimplexNoise.java

static float gradMap[16][3] = { { 1.0f, 1.0f, 0.0f },{ -1.0f, 1.0f, 0.0f },{ 1.0f, -1.0f, 0.0f },{ -1.0f, -1.0f, 0.0f },
	{ 1.0f, 0.0f, 1.0f },{ -1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, -1.0f },{ -1.0f, 0.0f, -1.0f },
	{ 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f, 1.0f },{ 0.0f, 1.0f, -1.0f },{ 0.0f, -1.0f, -1.0f } };

static unsigned int hashFunction(unsigned int seed)
{
	seed = (seed + 0x7ed55d16) + (seed << 12);
	seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
	seed = (seed + 0x165667b1) + (seed << 5);
	seed = (seed + 0xd3a2646c) ^ (seed << 9);
	seed = (seed + 0xfd7046c5) + (seed << 3);
	seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

	return seed;
}

static unsigned char calcPerm12(int p)
{
	return (unsigned char)(hashFunction(p) % 12);
}

static float dot(float g[3], float x, float y, float z) {
	return g[0] * x + g[1] * y + g[2] * z;
}

static float simplexNoise(float3 pos, float scale, int seed)
{
	float xin = pos.x * scale;
	float yin = pos.y * scale;
	float zin = pos.z * scale;

	float F3 = 1.0f / 3.0f;
	float G3 = 1.0f / 6.0f;

	float n0, n1, n2, n3;

	float s = (xin + yin + zin) * F3;
	int i = floorf(xin + s);
	int j = floorf(yin + s);
	int k = floorf(zin + s);
	float t = (i + j + k) * G3;
	float X0 = i - t;
	float Y0 = j - t;
	float Z0 = k - t;
	float x0 = xin - X0;
	float y0 = yin - Y0;
	float z0 = zin - Z0;

	int i1, j1, k1;
	int i2, j2, k2;
	if (x0 >= y0)
	{
		if (y0 >= z0)
		{
			i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f;
		}
		else if (x0 >= z0)
		{
			i1 = 1.0f;
			j1 = 0.0f;
			k1 = 0.0f;
			i2 = 1.0f;
			j2 = 0.0f;
			k2 = 1.0f;
		}
		else
		{
			i1 = 0.0f;
			j1 = 0.0f;
			k1 = 1.0f;
			i2 = 1.0f;
			j2 = 0.0f;
			k2 = 1.0f;
		}
	}
	else // x0<y0
	{
		if (y0 < z0)
		{
			i1 = 0.0f;
			j1 = 0.0f;
			k1 = 1.0f;
			i2 = 0.0f;
			j2 = 1;
			k2 = 1.0f;
		}
		else if (x0 < z0)
		{
			i1 = 0.0f;
			j1 = 1.0f;
			k1 = 0.0f;
			i2 = 0.0f;
			j2 = 1.0f;
			k2 = 1.0f;
		}
		else
		{
			i1 = 0.0f;
			j1 = 1.0f;
			k1 = 0.0f;
			i2 = 1.0f;
			j2 = 1.0f;
			k2 = 0.0f;
		}
	}

	float x1 = x0 - i1 + G3;
	float y1 = y0 - j1 + G3;
	float z1 = z0 - k1 + G3;
	float x2 = x0 - i2 + 2.0f * G3;
	float y2 = y0 - j2 + 2.0f * G3;
	float z2 = z0 - k2 + 2.0f * G3;
	float x3 = x0 - 1.0f + 3.0f * G3;
	float y3 = y0 - 1.0f + 3.0f * G3;
	float z3 = z0 - 1.0f + 3.0f * G3;

	int gi0 = calcPerm12(seed + (i * 607495) + (j * 359609) + (k * 654846));
	int gi1 = calcPerm12(seed + (i + i1) * 607495 + (j + j1) * 359609 + (k + k1) * 654846);
	int gi2 = calcPerm12(seed + (i + i2) * 607495 + (j + j2) * 359609 + (k + k2) * 654846);
	int gi3 = calcPerm12(seed + (i + 1) * 607495 + (j + 1) * 359609 + (k + 1) * 654846);

	float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
	if (t0 < 0.0f) n0 = 0.0f;
	else {
		t0 *= t0;
		n0 = t0 * t0 * dot(gradMap[gi0], x0, y0, z0);
	}
	float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
	if (t1 < 0.0f) n1 = 0.0f;
	else {
		t1 *= t1;
		n1 = t1 * t1 * dot(gradMap[gi1], x1, y1, z1);
	}
	float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
	if (t2 < 0.0f) n2 = 0.0f;
	else {
		t2 *= t2;
		n2 = t2 * t2 * dot(gradMap[gi2], x2, y2, z2);
	}
	float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
	if (t3 < 0.0f) n3 = 0.0f;
	else {
		t3 *= t3;
		n3 = t3 * t3 * dot(gradMap[gi3], x3, y3, z3);
	}

	return 32.0f * (n0 + n1 + n2 + n3);
}

static float repeaterSimplex(float3 pos, float scale, int seed, int n, float lacunarity, float decay)
{
	float acc = 0.0f;
	float amp = 1.0f;

	for (int i = 0; i < n; i++)
	{
		acc += simplexNoise(make_float3(pos.x, pos.y, pos.z), scale, seed) * amp * 0.35f;
		scale *= lacunarity;
		amp *= decay;
		seed = seed ^ ((i + 672381) * 200394);
	}

	return acc;
}

static int generate_terrain_height(int worldX, int worldZ)
{
	float simplexValue = repeaterSimplex(make_float3(static_cast<float>(worldX) / 64.0f, 0.0f, static_cast<float>(worldZ) / 64.0f), 1.0f, CONFIG_SEED_BASE, 3, 3.0f, 0.25f);
	simplexValue = ((simplexValue + 1.0f) / 2.0f);

	return (int)floorf(simplexValue * CONFIG_TERRAIN_HEIGHT) + 1;
}

static LONGLONG frequency;
static LARGE_INTEGER qpcCurrentTime;

static uint8_t* terrainData;

void cpu_generate_init()
{
	LARGE_INTEGER qpfFrequency;
	QueryPerformanceFrequency(&qpfFrequency);
	frequency = qpfFrequency.QuadPart;

	terrainData = (uint8_t*)malloc(CONFIG_TERRAIN_HEIGHT * sizeof(uint8_t));
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

}

void cpu_generate_clean()
{
	free(terrainData);
	terrainData = NULL;
}

static void gen_basic(uint8_t* blocks, int3 worldPos)
{
	for (int y = 0; y < CHUNK_SIZE; y++)
	{
		for (int x = 0; x < CHUNK_SIZE; x++)
		{
			for (int z = 0; z < CHUNK_SIZE; z++)
			{
				int worldX = x + worldPos.x;
				int worldY = y + worldPos.y;
				int worldZ = z + worldPos.z;

				int height = generate_terrain_height(worldX, worldZ);

				uint8_t block = 0;
				if (worldY == height) block = 3;
				else if (worldY < height && worldY > height - 3) block = 2;
				else if (worldY < height) block = 1;
				blocks[CHUNK_OFFSET(x, y, z)] = block;
			}
		}
	}
}

static void gen_const_mem(uint8_t* blocks, int3 worldPos)
{
	for (int y = 0; y < CHUNK_SIZE; y++)
	{
		for (int x = 0; x < CHUNK_SIZE; x++)
		{
			for (int z = 0; z < CHUNK_SIZE; z++)
			{
				int worldX = x + worldPos.x;
				int worldY = y + worldPos.y;
				int worldZ = z + worldPos.z;

				int height = generate_terrain_height(worldX, worldZ);

				int offset = height - worldY;
				blocks[CHUNK_OFFSET(x, y, z)] = terrainData[offset];
			}
		}
	}
}

static void gen_height_map(uint8_t* blocks, int3 worldPos)
{
	for (int x = 0; x < CHUNK_SIZE; x++)
	{
		for (int z = 0; z < CHUNK_SIZE; z++)
		{
			int worldX = x + worldPos.x;
			int worldZ = z + worldPos.z;
			int height = generate_terrain_height(worldX, worldZ);

			for (int y = 0; y < CHUNK_SIZE; y++)
			{
				int worldY = y + worldPos.y;

				int offset = height - worldY;
				blocks[CHUNK_OFFSET(x, y, z)] = terrainData[offset];
			}
		}
	}
}

void cpu_generate_chunk(Chunk* chunk)
{
	memset(chunk->blocks, 0, CHUNK_BLOCKS);

	int3 chunkPos = chunk->getChunkPosition();
	vec3 worldPosition = vec3(chunkPos.x, chunkPos.y, chunkPos.z) * (float)CHUNK_SIZE;
	int3 worldBlockPos = make_int3((int)worldPosition.x, (int)worldPosition.y, (int)worldPosition.z);

	LARGE_INTEGER qpcCurrentTime;
	QueryPerformanceCounter(&qpcCurrentTime);
	LONGLONG startTime = qpcCurrentTime.QuadPart;

#ifdef GEN_NAIV 
	gen_basic(chunk->blocks, worldBlockPos);
#endif

#ifdef GEN_CONST_MEM_SHARED_MEM
	gen_const_mem(chunk->blocks, worldBlockPos);
#endif

#ifdef GEN_HEIGHTMAP
	gen_height_map(chunk->blocks, worldBlockPos);
#endif

	LONGLONG currentTime;
	QueryPerformanceCounter(&qpcCurrentTime);
	currentTime = qpcCurrentTime.QuadPart;

	double elapsedTime = (static_cast<double>(currentTime - startTime) / frequency) * 1000.0;
	printf("Generate time: %f ms\n", elapsedTime);
}


