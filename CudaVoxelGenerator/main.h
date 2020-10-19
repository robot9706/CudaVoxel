#pragma once

#ifndef __MAIN__
#define __MAIN__

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

void gl_setup();
void gl_frame();

void main_loop();
void main_exit();

#endif