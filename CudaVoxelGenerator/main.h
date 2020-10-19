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

bool graphics_init();
void graphics_cleanup();
bool graphics_swap();
POINT graphics_size();

void gl_setup();
void gl_frame();
void gl_create_buffer(GLuint* vertexBuffer, GLuint* indexBuffer, float* vertexData, int numVertices, uint16_t* indexData, int numIndex);

void main_loop();
void main_exit();

#endif