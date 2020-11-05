#include "chunk.h"
#include "main.h"

#include "generator.h"

#include <vector>
using namespace std;

struct block_def {
	uint8_t textures[6]; //Top,Bottom,Left,Right,Forward,Backward
};

#define NUM_BLOCKS 7
block_def blocks[NUM_BLOCKS] = {
	{.textures = { 0 } }, // Air

	{.textures = { 0, 0, 0, 0, 0, 0 } }, // Stone
	{.textures = { 1, 1, 1, 1, 1, 1 } }, // Dirt
	{.textures = { 3, 1, 2, 2, 2, 2 } }, // Grass
	{.textures = { 4, 4, 4, 4, 4, 4 } }, // Sand
	{.textures = { 6, 6, 5, 5, 5, 5 } }, // Log
	{.textures = { 7, 7, 7, 7, 7, 7 } }, // Leaves
};

Chunk::Chunk(int3 chunkPosition)
{
	this->chunkPosition = chunkPosition;
	this->vbo = 0;
	this->ibo = 0;
	this->numRender = 0;
	this->geometryDirty = false;

	this->blocks = (uint8_t*)malloc(CHUNK_BLOCKS);
	memset(this->blocks, 0, CHUNK_BLOCKS);
}

Chunk::~Chunk()
{
	free(this->blocks);

	if (this->vbo > 0) {
		glDeleteBuffers(1, &this->vbo);
	}
	if (this->ibo > 0) {
		glDeleteBuffers(1, &this->ibo);
	}
}

int3 Chunk::getChunkPosition()
{
	return this->chunkPosition;
}

static void get_block_uv(int textureIndex, float* uvs)
{
	int texX = textureIndex % TEXTURE_SIZE;
	int texY = textureIndex / TEXTURE_SIZE;

	float cellSize = 1.0f / TEXTURE_SIZE;

	// Top left
	uvs[0] = texX * cellSize;
	uvs[1] = texY * cellSize;

	// Top right
	uvs[2] = (texX + 1) * cellSize;
	uvs[3] = texY * cellSize;

	// Bottom left
	uvs[4] = texX * cellSize;
	uvs[5] = (texY + 1) * cellSize;

	// Bottom right
	uvs[6] = (texX + 1) * cellSize;
	uvs[7] = (texY + 1) * cellSize;
}

static bool is_transparent(Chunk* chunk, int x, int y, int z)
{
	if (x < 0 || y < 0 || z < 0 || x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE)
		return true;

	uint8_t blockAt = chunk->blocks[CHUNK_OFFSET(x, y, z)];
	return (blockAt == BLOCK_AIR || blockAt == BLOCK_LEAVES);
}

static void build_block(Chunk* chunk, uint8_t id, int blockX, int blockY, float blockZ, vector<float> *vertices, vector<INDEX_TYPE> *indices)
{
	block_def* block = &blocks[id];

	INDEX_TYPE baseIndex;
	float uvs[8];

	float x = blockX;
	float y = blockY;
	float z = blockZ;
	
	// Top
	if (is_transparent(chunk, x, y + 1, z)) 
	{
		baseIndex = (uint32_t)(vertices->size() / 5);

		get_block_uv(block->textures[0], uvs);

		vertices->insert(vertices->end(), { x,     y + 1, z,     uvs[0], uvs[1] });
		vertices->insert(vertices->end(), { x + 1, y + 1, z,     uvs[2], uvs[3] });
		vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[6], uvs[7] });
		vertices->insert(vertices->end(), { x,     y + 1, z + 1, uvs[4], uvs[5] });

		indices->insert(indices->end(), { baseIndex, (INDEX_TYPE)(baseIndex + 1), (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 3), baseIndex });
	}

	// Bottom
	if (is_transparent(chunk, x, y - 1, z))
	{
		baseIndex = (uint32_t)(vertices->size() / 5);

		get_block_uv(block->textures[1], uvs);

		vertices->insert(vertices->end(), { x,     y, z,     uvs[0], uvs[1] });
		vertices->insert(vertices->end(), { x + 1, y, z,     uvs[2], uvs[3] });
		vertices->insert(vertices->end(), { x + 1, y, z + 1, uvs[6], uvs[7] });
		vertices->insert(vertices->end(), { x,     y, z + 1, uvs[4], uvs[5] });

		indices->insert(indices->end(), { (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 1), baseIndex, baseIndex, (INDEX_TYPE)(baseIndex + 3), (INDEX_TYPE)(baseIndex + 2) });
	}

	// Right
	if (is_transparent(chunk, x + 1, y, z))
	{
		baseIndex = (uint32_t)(vertices->size() / 5);

		get_block_uv(block->textures[3], uvs);

		vertices->insert(vertices->end(), { x + 1, y,     z,     uvs[4], uvs[5] });
		vertices->insert(vertices->end(), { x + 1, y + 1, z,     uvs[0], uvs[1] });
		vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[2], uvs[3] });
		vertices->insert(vertices->end(), { x + 1, y,     z + 1, uvs[6], uvs[7] });

		indices->insert(indices->end(), { (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 1), baseIndex, baseIndex, (INDEX_TYPE)(baseIndex + 3), (INDEX_TYPE)(baseIndex + 2) });
	}

	// Left
	if (is_transparent(chunk, x - 1, y, z))
	{
		baseIndex = (uint32_t)(vertices->size() / 5);

		get_block_uv(block->textures[2], uvs);

		vertices->insert(vertices->end(), { x, y,     z,     uvs[6], uvs[7] });
		vertices->insert(vertices->end(), { x, y + 1, z,     uvs[2], uvs[3] });
		vertices->insert(vertices->end(), { x, y + 1, z + 1, uvs[0], uvs[1] });
		vertices->insert(vertices->end(), { x, y,     z + 1, uvs[4], uvs[5] });

		indices->insert(indices->end(), { baseIndex, (INDEX_TYPE)(baseIndex + 1), (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 3), baseIndex });
	}

	// Back
	if (is_transparent(chunk, x, y, z + 1))
	{
		baseIndex = (uint32_t)(vertices->size() / 5);

		get_block_uv(block->textures[5], uvs);

		vertices->insert(vertices->end(), { x,     y,     z + 1, uvs[4], uvs[5] });
		vertices->insert(vertices->end(), { x + 1, y,     z + 1, uvs[6], uvs[7] });
		vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[2], uvs[3] });
		vertices->insert(vertices->end(), { x,     y + 1, z + 1, uvs[0], uvs[1] });

		indices->insert(indices->end(), { (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 1), baseIndex, baseIndex, (INDEX_TYPE)(baseIndex + 3), (INDEX_TYPE)(baseIndex + 2) });
	}

	// Front
	if (is_transparent(chunk, x, y, z - 1))
	{
		baseIndex = (uint32_t)(vertices->size() / 5);

		get_block_uv(block->textures[4], uvs);

		vertices->insert(vertices->end(), { x,     y,     z, uvs[4], uvs[5] });
		vertices->insert(vertices->end(), { x + 1, y,     z, uvs[6], uvs[7] });
		vertices->insert(vertices->end(), { x + 1, y + 1, z, uvs[2], uvs[3] });
		vertices->insert(vertices->end(), { x,     y + 1, z, uvs[0], uvs[1] });

		indices->insert(indices->end(), { baseIndex, (INDEX_TYPE)(baseIndex + 1), (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 2), (INDEX_TYPE)(baseIndex + 3), baseIndex });
	}
}

void Chunk::generate()
{
	// Generate blocks
	cuda_generate_chunk(this);

	// Build geometry
	for (int x = 0; x < CHUNK_SIZE; x++)
	{
		for (int y = 0; y < CHUNK_SIZE; y++)
		{
			for (int z = 0; z < CHUNK_SIZE; z++)
			{
				int offset = CHUNK_OFFSET(x, y, z);
				if (blocks[offset] == 0)
					continue;

				build_block(this, blocks[offset], x, y, z, &vertices, &indices);
			}
		}
	}

	if (vertices.size() == 0) {
		this->numRender = 0;
		this->geometryDirty = false;
		return;
	}

	this->geometryDirty = true;
}

bool Chunk::upload()
{
	if (!this->geometryDirty) 
	{
		return false;
	}

	gl_create_buffer(&this->vbo, &this->ibo, vertices.data(), vertices.size(), indices.data(), indices.size() * sizeof(INDEX_TYPE));
	this->numRender = indices.size();

	vertices.clear();
	indices.clear();

	this->geometryDirty = false;

	return true;
}

void Chunk::render()
{
	if (this->vbo == 0 || this->ibo == 0 || this->numRender == 0)
	{
		return;
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)0); //XYZ--
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3)); //---UV

	glDrawElements(GL_TRIANGLES, this->numRender, INDEX_TYPE_GL, NULL);
}