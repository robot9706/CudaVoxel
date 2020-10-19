#pragma once

#ifndef __CHUNK__
#define __CHUNK__

#define TEXTURE_SIZE 16

#define CHUNK_SIZE 16
#define CHUNK_BLOCKS (CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE)

#include "gl.h"

#include <stdint.h>
using namespace std;

class Chunk {
private:
	uint8_t* blocks;

	vec3 chunkPosition;

	GLuint vbo;
	GLuint ibo;

	int numRender;

public:
	Chunk(vec3 chunkPosition);
	~Chunk();

	vec3 getChunkPosition();

	void generate();
	void render();
};

#endif
