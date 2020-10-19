#include "chunk.h"
#include "main.h"

#include <vector>
using namespace std;

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

static void build_block(uint8_t id, float x, float y, float z, vector<float> *vertices, vector<uint16_t> *indices)
{
	// TODO: Cuda?

	uint16_t baseIndex;
	float uvs[8];
	
	// Top
	baseIndex = (uint16_t)vertices->size();

	get_block_uv(2, uvs);

	vertices->insert(vertices->end(), { x,     y + 1, z,     uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z,     uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x,     y + 1, z + 1, uvs[4], uvs[5] });

	indices->insert(indices->end(), { baseIndex, (uint16_t)(baseIndex + 1), (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 3), baseIndex });

	// Bottom
	baseIndex = (uint16_t)(vertices->size() / 5);

	vertices->insert(vertices->end(), { x,     y, z,     uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x + 1, y, z,     uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x + 1, y, z + 1, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x,     y, z + 1, uvs[4], uvs[5] });

	indices->insert(indices->end(), { (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 1), baseIndex, baseIndex, (uint16_t)(baseIndex + 3), (uint16_t)(baseIndex + 2) });

	// Right
	baseIndex = (uint16_t)(vertices->size() / 5);

	vertices->insert(vertices->end(), { x + 1, y,     z,     uvs[4], uvs[5] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z,     uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x + 1, y,     z + 1, uvs[6], uvs[7] });
		
	indices->insert(indices->end(), { (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 1), baseIndex, baseIndex, (uint16_t)(baseIndex + 3), (uint16_t)(baseIndex + 2) });

	// Left
	baseIndex = (uint16_t)(vertices->size() / 5);

	vertices->insert(vertices->end(), { x, y,     z,     uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x, y + 1, z,     uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x, y + 1, z + 1, uvs[0], uvs[1] });
	vertices->insert(vertices->end(), { x, y,     z + 1, uvs[4], uvs[5] });
		
	indices->insert(indices->end(), { baseIndex, (uint16_t)(baseIndex + 1), (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 3), baseIndex });

	// Back
	baseIndex = (uint16_t)(vertices->size() / 5);

	vertices->insert(vertices->end(), { x,     y,     z + 1, uvs[4], uvs[5] });
	vertices->insert(vertices->end(), { x + 1, y,     z + 1, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z + 1, uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x,     y + 1, z + 1, uvs[0], uvs[1] });

	indices->insert(indices->end(), { (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 1), baseIndex, baseIndex, (uint16_t)(baseIndex + 3), (uint16_t)(baseIndex + 2) });

	// Front
	baseIndex = (uint16_t)(vertices->size() / 5);

	vertices->insert(vertices->end(), { x,     y,     z, uvs[4], uvs[5] });
	vertices->insert(vertices->end(), { x + 1, y,     z, uvs[6], uvs[7] });
	vertices->insert(vertices->end(), { x + 1, y + 1, z, uvs[2], uvs[3] });
	vertices->insert(vertices->end(), { x,     y + 1, z, uvs[0], uvs[1] });

	indices->insert(indices->end(), { baseIndex, (uint16_t)(baseIndex + 1), (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 2), (uint16_t)(baseIndex + 3), baseIndex });
}

void Chunk::generate()
{
	// Generate blocks

	// Build geometry
	vector<float> vertices;
	vector<uint16_t> indices;

	// TODO: All blocks
	build_block(0, 0, 0, 0, &vertices, &indices);

	gl_create_buffer(&this->vbo, &this->ibo, vertices.data(), vertices.size(), indices.data(), indices.size());
	
	this->numRender = indices.size();
}

void Chunk::render()
{
	if (this->vbo == 0 || this->ibo == 0) 
	{
		return;
	}

	glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ibo);

	glDrawElements(GL_TRIANGLES, this->numRender, GL_UNSIGNED_SHORT, NULL);
}