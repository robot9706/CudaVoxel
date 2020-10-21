#pragma once

#ifndef __GENERATOR__
#define __GENERATOR__

#include "cuda_runtime.h"
#include "chunk.h"

void cuda_generate_init();
void cuda_generate_clean();

void cuda_generate_chunk(Chunk* chunk);

#endif
