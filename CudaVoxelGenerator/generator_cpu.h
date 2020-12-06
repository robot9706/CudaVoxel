#pragma once

#ifndef __GEN_CPU__
#define __GEN_CPU__

#include "cuda_runtime.h"
#include "chunk.h"

void cpu_generate_init();
void cpu_generate_clean();

void cpu_generate_chunk(Chunk* chunk);

#endif