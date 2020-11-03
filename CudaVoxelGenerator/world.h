#pragma once

#ifndef __WORLD__
#define __WORLD__

#include <map>
#include <iostream>
#include <cassert>

#include "gl.h"
#include "chunk.h"

#include <Windows.h>

using namespace std;

class World
{
private:
	HANDLE generatorThread;

public:
	map<uint64_t, Chunk**> regions;
	HANDLE generatorSignal;
	bool generateChunks;
	int3 lastCenterChunk;

	World();
	~World();

	void insertChunk(Chunk* chunk);
	bool hasChunk(int3 pos);

	void start();
	void stop();

	void render(vec3 center);
};

#endif