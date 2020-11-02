#include "chunk.h"
#include "main.h"

#include "generator.h"

#include <vector>
using namespace std;

struct block_def {
	uint8_t textures[6]; //Top,Bottom,Left,Right,Forward,Backward
};

#define NUM_BLOCKS 5
block_def blocks[NUM_BLOCKS] = {
	{.textures = { 0 } }, // Air

	{.textures = { 0, 0, 0, 0, 0, 0 } }, // Stone
	{.textures = { 1, 1, 1, 1, 1, 1 } }, // Dirt
	{.textures = { 3, 1, 2, 2, 2, 2 } }, // Grass
	{.textures = { 4, 4, 4, 4, 4, 4 } } // Sand
};

Chunk::Chunk(vec3 chunkPosition)
{
	this->chunkPosition = chunkPosition;
	this->vbo = 0;
	this->ibo = 0;
	this->numRender = 0;

	this->blocks = (uint8_t*)malloc(CHUNK_BLOCKS);
	memset(this->blocks, 0, CHUNK_BLOCKS);
}

Chunk::~Chunk()
{
	free(this->blocks);
}

vec3 Chunk::getChunkPosition()
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

static void build_block(uint8_t id, float x, float y, float z, vector<float> *vertices, vector<uint32_t> *indices)
{
	block_def* block = &blocks[id];

	uint32_t baseIndex;
	float uvs[8];
	
	// Top
	baseIndex = (uint32_t)(vertices->size() / 5);

	get_block_uv(block->textures[0], uvs);

	vertices->insert(vertices->end(), { x,     y + 1, z,     uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z,     uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x,     y + 1, z + 1, uvs[4], uvs[5] });

	indices->insert(indices->end(), { baseIndex, (uint32_t)(baseIndex + 1), (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 3), baseIndex });

	// Bottom
	baseIndex = (uint32_t)(vertices->size() / 5);

	get_block_uv(block->textures[1], uvs);

	vertices->insert(vertices->end(), { x,     y, z,     uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x + 1, y, z,     uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x + 1, y, z + 1, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x,     y, z + 1, uvs[4], uvs[5] });

	indices->insert(indices->end(), { (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 1), baseIndex, baseIndex, (uint32_t)(baseIndex + 3), (uint32_t)(baseIndex + 2) });

	// Right
	baseIndex = (uint32_t)(vertices->size() / 5);

	get_block_uv(block->textures[3], uvs);

	vertices->insert(vertices->end(), { x + 1, y,     z,     uvs[4], uvs[5] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z,     uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x + 1, y,     z + 1, uvs[6], uvs[7] });
		
	indices->insert(indices->end(), { (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 1), baseIndex, baseIndex, (uint32_t)(baseIndex + 3), (uint32_t)(baseIndex + 2) });

	// Left
	baseIndex = (uint32_t)(vertices->size() / 5);

	get_block_uv(block->textures[2], uvs);

	vertices->insert(vertices->end(), { x, y,     z,     uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x, y + 1, z,     uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x, y + 1, z + 1, uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x, y,     z + 1, uvs[4], uvs[5] });
		
	indices->insert(indices->end(), { baseIndex, (uint32_t)(baseIndex + 1), (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 3), baseIndex });

	// Back
	baseIndex = (uint32_t)(vertices->size() / 5);

	get_block_uv(block->textures[5], uvs);

	vertices->insert(vertices->end(), { x,     y,     z + 1, uvs[4], uvs[5] });
	vertices->insert(vertices->end(), { x + 1, y,     z + 1, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x,     y + 1, z + 1, uvs[0], uvs[1] });

	indices->insert(indices->end(), { (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 1), baseIndex, baseIndex, (uint32_t)(baseIndex + 3), (uint32_t)(baseIndex + 2) });

	// Front
	baseIndex = (uint32_t)(vertices->size() / 5);

	get_block_uv(block->textures[4], uvs);

	vertices->insert(vertices->end(), { x,     y,     z, uvs[4], uvs[5] });
	vertices->insert(vertices->end(), { x + 1, y,     z, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z, uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x,     y + 1, z, uvs[0], uvs[1] });

	indices->insert(indices->end(), { baseIndex, (uint32_t)(baseIndex + 1), (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 2), (uint32_t)(baseIndex + 3), baseIndex });
}

void Chunk::generate()
{
	// Generate blocks
	cuda_generate_chunk(this);

	// Build geometry
	vector<float> vertices;
	vector<uint32_t> indices;

	for (int x = 0; x < CHUNK_SIZE; x++)
	{
		for (int y = 0; y < CHUNK_SIZE; y++)
		{
			for (int z = 0; z < CHUNK_SIZE; z++)
			{
				int offset = CHUNK_OFFSET(x, y, z);
				if (blocks[offset] == 0)
					continue;

				build_block(blocks[offset], x, y, z, &vertices, &indices);
			}
		}
	}

	gl_create_buffer(&this->vbo, &this->ibo, vertices.data(), vertices.size(), indices.data(), indices.size());
	
	this->numRender = indices.size();
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

	glDrawElements(GL_TRIANGLES, this->numRender, GL_UNSIGNED_INT, NULL);
}