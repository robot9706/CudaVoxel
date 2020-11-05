#pragma once

#ifndef __CHUNK__
#define __CHUNK__

#define BLOCK_AIR 0
#define BLOCK_STONE 1
#define BLOCK_DIRT 2
#define BLOCK_GRASS 3
#define BLOCK_SAND 4
#define BLOCK_LOG 5
#define BLOCK_LEAVES 6

#define TEXTURE_SIZE 16

#define CHUNK_SIZE 16
#define CHUNK_BLOCKS (CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE)

#define CHUNK_OFFSET(X,Y,Z) ((Z)*CHUNK_SIZE*CHUNK_SIZE+(Y)*CHUNK_SIZE+(X))

#define INDEX_TYPE uint16_t
#define INDEX_TYPE_GL GL_UNSIGNED_SHORT

#include "gl.h"
#include "cuda_runtime.h"

#include <vector>
#include <stdint.h>
using namespace std;

class Chunk {
private:
	int3 chunkPosition;

	GLuint vbo;
	GLuint ibo;

	int numRender;

	bool geometryDirty;
	vector<float> vertices;
	vector<INDEX_TYPE> indices;

public:
	uint8_t* blocks;

	Chunk(int3 chunkPosition);
	~Chunk();

	int3 getChunkPosition();

	void generate();
	bool upload();
	void render();
};

#endif
