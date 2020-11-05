#pragma once

#ifndef __MAIN__
#define __MAIN__

#include "gl.h"

#include <Windows.h>

#include <string>
#include <memory>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <stdbool.h>

using namespace std;

#include "chunk.h"

bool graphics_init();
void graphics_cleanup();
bool graphics_swap();
POINT graphics_size();

void gl_setup();
void gl_cleanup();
void gl_frame(float dt);
void gl_create_buffer(GLuint* vertexBuffer, GLuint* indexBuffer, float* vertexData, int numVertices, void* indexData, int indexSize);
void gl_render_chunk(Chunk* chunk);

void main_loop();
void main_exit();

bool keyboard_check(int key);

#endif