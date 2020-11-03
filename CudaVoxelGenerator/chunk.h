#pragma once

#ifndef __CHUNK__
#define __CHUNK__

#define TEXTURE_SIZE 16

#define CHUNK_SIZE 16
#define CHUNK_BLOCKS (CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE)

#define CHUNK_OFFSET(X,Y,Z) (Z*CHUNK_SIZE*CHUNK_SIZE+Y*CHUNK_SIZE+X)

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
	vector<uint32_t> indices;

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
