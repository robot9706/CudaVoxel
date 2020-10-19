#include "main.h"
#include "gl.h"

#include <Windows.h>

static bool run = false;

void main_loop()
{
	//Timing
	LONGLONG startTime;
	LONGLONG currentTime;
	LONGLONG frequency;
	double prevElapsedTime = 0.0;

	//Query high precision timer
	LARGE_INTEGER qpfFrequency;
	QueryPerformanceFrequency(&qpfFrequency);
	frequency = qpfFrequency.QuadPart;

	LARGE_INTEGER qpcCurrentTime;
	QueryPerformanceCounter(&qpcCurrentTime);
	startTime = qpcCurrentTime.QuadPart;

	//Loop
	run = true;

	float runTime = 0.0f;

	while (run)
	{
		//Query the timer
		QueryPerformanceCounter(&qpcCurrentTime);
		currentTime = qpcCurrentTime.QuadPart;

		//Calculat the delta time
		double elapsedTime = static_cast<double>(currentTime - startTime) / frequency;
		float deltaTime = (float)(elapsedTime - prevElapsedTime);
		runTime += deltaTime;

		//Call the runtime to do a game loop
		gl_frame();

		//Present the new frame
		if (!graphics_swap())
		{
			break;
		}

		//Process Win32 messages
		MSG msg;
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		//Time keeping
		prevElapsedTime = elapsedTime;
	}
}

void main_exit()
{
	run = false;
}