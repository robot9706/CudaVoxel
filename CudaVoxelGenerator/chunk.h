#pragma once

#ifndef __CHUNK__
#define __CHUNK__

#define TEXTURE_SIZE 16

#define CHUNK_SIZE 16
#define CHUNK_BLOCKS (CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE)

#define CHUNK_OFFSET(X,Y,Z) (Z*CHUNK_SIZE*CHUNK_SIZE+Y*CHUNK_SIZE+X)

#include "gl.h"

#include <stdint.h>
using namespace std;

class Chunk {
private:
	vec3 chunkPosition;

	GLuint vbo;
	GLuint ibo;

	int numRender;

public:
	uint8_t* blocks;

	Chunk(vec3 chunkPosition);
	~Chunk();

	vec3 getChunkPosition();

	void generate();
	void render();
};

#endif
