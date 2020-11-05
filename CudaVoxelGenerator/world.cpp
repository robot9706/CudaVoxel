#include "world.h"

#define VIEW_DIST_XZ 6
#define VIEW_DIST_Y 3

#include "generator.h"
#include "main.h"
#include "log.h"

#define REGION_SIZE 16
#define REGION_LENGTH (REGION_SIZE*REGION_SIZE*REGION_SIZE)

#define CLAMP(X) (X<0?0:X)

static uint64_t int3_hash(int x, int y, int z)
{
	return ((uint64_t)27 ^ (((uint64_t)x << 16) | ((uint64_t)y << 8) | (uint64_t)z));
}

static uint64_t int3_hash(int3 pos) 
{
	return int3_hash(pos.x, pos.y, pos.z);
}

static int3 chunk_to_region(int3 chunkPos)
{
	return make_int3((int)floorf((float)chunkPos.x / REGION_SIZE), (int)floorf((float)chunkPos.y / REGION_SIZE), (int)floorf((float)chunkPos.z / REGION_SIZE));
}

static int in_region_offset(int3 chunk) 
{
	int localX = chunk.x % REGION_SIZE;
	int localY = chunk.y % REGION_SIZE;
	int localZ = chunk.z % REGION_SIZE;

	if (localX < 0)
		localX = REGION_SIZE + localX;
	if (localY < 0)
		localY = REGION_SIZE + localY;
	if (localZ < 0)
		localZ = REGION_SIZE + localZ;

	return localX + localY * REGION_SIZE + localZ * REGION_SIZE * REGION_SIZE;
}

static DWORD WINAPI thread_world_create(LPVOID lpThreadParameter)
{
	CUDA_CHECK(cudaSetDevice(0));

	World* world = (World*)lpThreadParameter;
	while (world->generateChunks)
	{
		WaitForSingleObject(world->generatorSignal, INFINITE);

		int numGenerated = 0;

		int3 centerChunk = world->lastCenterChunk;
		for (int centerX = CLAMP(centerChunk.x - VIEW_DIST_XZ); centerX < CLAMP(centerChunk.x + VIEW_DIST_XZ); centerX++)
		{
			for (int centerZ = CLAMP(centerChunk.z - VIEW_DIST_XZ); centerZ < CLAMP(centerChunk.z + VIEW_DIST_XZ); centerZ++)
			{
				for (int centerY = CLAMP(centerChunk.y - VIEW_DIST_Y); centerY < CLAMP(centerChunk.y + VIEW_DIST_Y); centerY++)
				{
					if (!world->generateChunks) 
					{
						return 0;
					}

					if (world->hasChunk(make_int3(centerX, centerY, centerZ)))
						continue;

					numGenerated++;

					Chunk* newChunk = new Chunk(make_int3(centerX, centerY, centerZ));
					newChunk->generate();
					world->insertChunk(newChunk);
				}
			}
		}

		if (numGenerated == 0)
		{
			ResetEvent(world->generatorSignal);
		}
	}

	return 0;
}

World::World()
{
	this->lastCenterChunk = make_int3(0, 0, 0);
	this->generateChunks = true;
	this->generatorSignal = CreateEvent(NULL, true, true, TEXT("GeneratorSignal"));
	this->regionsMutex = CreateMutex(NULL, false, NULL);

	float size = (VIEW_DIST_XZ + 1) * CHUNK_SIZE;
	this->removeDistance = sqrtf(size * size + size * size);
}

World::~World()
{
	for (map<uint64_t, Chunk**>::iterator it = this->regions.begin(); it != this->regions.end(); ++it) {
		for (int x = 0; x < REGION_LENGTH; x++) {
			delete it->second[x];
		}
		delete [] it->second;
	}

	this->regions.clear();
}

void World::insertChunk(Chunk* chunk)
{
	WaitForSingleObject(this->regionsMutex, INFINITE);

	int3 chunkPos = chunk->getChunkPosition();
	int3 regionPos = chunk_to_region(chunkPos);

	uint64_t regionHash = int3_hash(regionPos);

	if (this->regions.contains(regionHash))
	{
		this->regions.find(regionHash)->second[in_region_offset(chunkPos)] = chunk;
	}
	else
	{
		Chunk** region = new Chunk*[REGION_LENGTH];
		memset(region, 0, sizeof(Chunk*) * REGION_LENGTH);
		region[in_region_offset(chunkPos)] = chunk;
		this->regions.insert(pair<uint64_t, Chunk**>(regionHash, region));
	}

	ReleaseMutex(this->regionsMutex);
}

bool World::hasChunk(int3 chunkPos)
{
	int3 regionPos = chunk_to_region(chunkPos);
	uint64_t regionHash = int3_hash(regionPos);

	if (!this->regions.contains(regionHash)) 
	{
		return false;
	}

	return (this->regions.find(regionHash)->second[in_region_offset(chunkPos)] != NULL);
}

void World::start()
{
	DWORD threadID;
	this->generatorThread = CreateThread(NULL, 0, &thread_world_create, this, 0, &threadID);
}

void World::stop()
{
	this->generateChunks = false;
	SetEvent(this->generatorSignal);
	CloseHandle(this->generatorThread);
	CloseHandle(this->regionsMutex);

	for (map<uint64_t, Chunk**>::iterator it = this->regions.begin(); it != this->regions.end(); ++it) {
		for (int x = 0; x < REGION_LENGTH; x++) {
			delete it->second[x];
		}
		delete[] it->second;
	}

	this->regions.clear();
}

void World::render(vec3 center)
{
	WaitForSingleObject(this->regionsMutex, INFINITE);

	int3 centerChunk = make_int3((int)floorf(center.x / CHUNK_SIZE), 0, (int)floorf(center.z / CHUNK_SIZE));

	if (this->lastCenterChunk.x != centerChunk.x || this->lastCenterChunk.y != centerChunk.y || this->lastCenterChunk.z != centerChunk.z) {
		this->lastCenterChunk = centerChunk;
		SetEvent(this->generatorSignal);

		vector<uint64_t> toRemove;
		for (map<uint64_t, Chunk**>::iterator it = this->regions.begin(); it != this->regions.end(); ++it) {
			Chunk** region = it->second;
			int numUsed = 0;
			for (int x = 0; x < REGION_LENGTH; x++) {
				if (region[x] == NULL) {
					continue;
				}

				int3 chunkPosition = region[x]->getChunkPosition();
				vec2 chunkWorldPos = vec2(chunkPosition.x, chunkPosition.z) * vec2(CHUNK_SIZE, CHUNK_SIZE);
				
				float dist = distance(chunkWorldPos, vec2(center.x, center.z));
				if (dist >= this->removeDistance) {
					delete region[x];
					region[x] = NULL;
				}
				else {
					numUsed++;
				}
			}

			if (numUsed == 0) {
				delete[] region;
				toRemove.push_back(it->first);
			}
		}

		for (int x = 0; x < toRemove.size(); x++) {
			this->regions.erase(toRemove[x]);
		}
	}

	if (this->regions.empty()) {
		ReleaseMutex(this->regionsMutex);
		return;
	}

	int maxUpload = 4;

	Chunk** currentRegion = NULL;
	uint64_t currentRegionHash;

	for (int centerY = 0; centerY < CLAMP(centerChunk.y + VIEW_DIST_Y); centerY++)
	{
		for (int centerX = CLAMP(centerChunk.x - VIEW_DIST_XZ); centerX < CLAMP(centerChunk.x + VIEW_DIST_XZ); centerX++)
		{
			for (int centerZ = CLAMP(centerChunk.z - VIEW_DIST_XZ); centerZ < CLAMP(centerChunk.z + VIEW_DIST_XZ); centerZ++)
			{
				int3 chunkPos = make_int3(centerX, centerY, centerZ);
				int3 regionPos = chunk_to_region(chunkPos);
				uint64_t regionHash = int3_hash(regionPos);

				if (currentRegion == NULL || currentRegionHash != regionHash)
				{
					if (this->regions.contains(regionHash))
					{
						currentRegion = this->regions.find(regionHash)->second;
						currentRegionHash = regionHash;
					}
				}

				int offset = in_region_offset(chunkPos);
				if (currentRegion == NULL || currentRegion[offset] == NULL)
				{
					continue;
				}

				Chunk* chunk = currentRegion[offset];

				if (maxUpload > 0)
				{
					if (chunk->upload())
					{
						maxUpload--;
					}
				}

				gl_render_chunk(chunk);
			}
		}
	}

	ReleaseMutex(this->regionsMutex);
}