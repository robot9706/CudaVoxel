
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>

using namespace std;

#include <Windows.h>
#include <windowsx.h>

#define GL_GLEXT_PROTOTYPES

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>

using namespace glm;

#include <memory>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <stdbool.h>

#define ERROR_FORMAT(FORMAT, PARAMS) fprintf(stderr, FORMAT, PARAMS); exit(-1);
#define ERROR_TEXT(TEXT) ERROR_FORMAT("%s\n", TEXT)

static HDC eglHDC;
static EGLNativeWindowType eglWindow;
static EGLDisplay eglDisplay;
static EGLContext eglContext;
static EGLSurface eglSurface;

static bool run = false;

static wstring eglWindowClass = L"CudaVoxel";

static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		//Destroy / close
		case WM_DESTROY:
		case WM_CLOSE:
		{
			run = false;
			break;
		}

		//Window resize
		case WM_SIZE:
		{
			RECT winRect;
			GetClientRect(hWnd, &winRect);

			POINT topLeft;
			topLeft.x = winRect.left;
			topLeft.y = winRect.top;
			ClientToScreen(hWnd, &topLeft);

			POINT botRight;
			botRight.x = winRect.right;
			botRight.y = winRect.bottom;
			ClientToScreen(hWnd, &botRight);

			//MainResolutionChanged((uint16_t)(botRight.x - topLeft.x), (uint16_t)(botRight.y - topLeft.y));

			break;
		}
	}

	return DefWindowProcW(hWnd, message, wParam, lParam);
}

bool egl_window_create(LONG width, LONG height, wstring title)
{
	WNDCLASSEXW windowClass = { 0 };
	windowClass.cbSize = sizeof(WNDCLASSEXW);
	windowClass.style = CS_OWNDC;
	windowClass.lpfnWndProc = WndProc;
	windowClass.cbClsExtra = 0;
	windowClass.cbWndExtra = 0;
	windowClass.hInstance = GetModuleHandle(NULL);
	windowClass.hIcon = NULL;
	windowClass.hCursor = LoadCursorA(NULL, IDC_ARROW);
	windowClass.hbrBackground = 0;
	windowClass.lpszMenuName = NULL;
	windowClass.lpszClassName = eglWindowClass.c_str();
	if (!RegisterClassExW(&windowClass))
	{
		return false;
	}

	DWORD style = (WS_CAPTION | WS_MINIMIZEBOX | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_SYSMENU);
	DWORD extendedStyle = WS_EX_APPWINDOW;

	RECT sizeRect = { 0, 0, width, height };
	AdjustWindowRectEx(&sizeRect, style, false, extendedStyle);

	width = sizeRect.right - sizeRect.left;
	height = sizeRect.bottom - sizeRect.top;

	//Create the actual window
	eglWindow = CreateWindowExW(extendedStyle, eglWindowClass.c_str(), title.c_str(), style, CW_USEDEFAULT, CW_USEDEFAULT,
		width, height, NULL, NULL,
		GetModuleHandle(NULL), nullptr);

	//Center the window
	HWND   hwndScreen;
	RECT   rectScreen;
	hwndScreen = GetDesktopWindow();
	GetWindowRect(hwndScreen, &rectScreen);

	int posX = ((rectScreen.right - rectScreen.left) / 2 - (width / 2));
	int posY = ((rectScreen.bottom - rectScreen.top) / 2 - (height / 2));
	SetWindowPos(eglWindow, NULL, posX, posY, 0, 0, SWP_SHOWWINDOW | SWP_NOSIZE);

	//Get the handle
	eglHDC = GetDC(eglWindow);
	if (!eglHDC)
	{
		return false;
	}

	return true;
}

void egl_window_cleanup()
{
	if (eglHDC)
	{
		ReleaseDC(eglWindow, eglHDC);
		eglHDC = 0;
	}

	if (eglWindow)
	{
		DestroyWindow(eglWindow);
		eglWindow = 0;
	}

	UnregisterClassW(eglWindowClass.c_str(), NULL);
}

bool egl_init()
{
	const EGLint configAttributes[] =
	{
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 16,
		EGL_STENCIL_SIZE, 0,
		EGL_NONE
	};

	const EGLint surfaceAttributes[] =
	{
		EGL_NONE
	};

	const EGLint contextAttibutes[] =
	{
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};

	const EGLint displayAttributes[] =
	{
		EGL_PLATFORM_ANGLE_TYPE_ANGLE, EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE,
		EGL_NONE,
	};

	EGLConfig config = 0;

	// ANGLE: eglGetPlatformDisplayEXT is an alternative to eglGetDisplay. It allows us to specifically request D3D11 instead of D3D9.
	PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
	if (!eglGetPlatformDisplayEXT)
	{
		ERROR_TEXT("Failed to get function eglGetPlatformDisplayEXT");
	}

	eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, eglHDC, displayAttributes);
	if (eglDisplay == EGL_NO_DISPLAY)
	{
		ERROR_TEXT("Failed to get requested EGL display");
		return false;
	}

	if (eglInitialize(eglDisplay, NULL, NULL) == EGL_FALSE)
	{
		ERROR_TEXT("Failed to initialize EGL");
		return false;
	}

	EGLint numConfigs;
	if ((eglChooseConfig(eglDisplay, configAttributes, &config, 1, &numConfigs) == EGL_FALSE) || (numConfigs == 0))
	{
		ERROR_TEXT("Failed to choose first EGLConfig");
		return false;
	}

	eglSurface = eglCreateWindowSurface(eglDisplay, config, eglWindow, surfaceAttributes);
	if (eglSurface == EGL_NO_SURFACE)
	{
		ERROR_TEXT("Failed to create EGL fullscreen surface");
		return false;
	}

	if (eglGetError() != EGL_SUCCESS)
	{
		ERROR_TEXT("eglGetError has reported an error");
		return false;
	}

	eglContext = eglCreateContext(eglDisplay, config, NULL, contextAttibutes);
	if (eglContext == EGL_NO_CONTEXT)
	{
		ERROR_TEXT("Failed to create EGL context");
		return false;
	}

	if (eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext) == EGL_FALSE)
	{
		ERROR_TEXT("Failed to make EGLSurface current");
		return false;
	}

	return true;
}

const char* color_shader_vs = "precision mediump float;"
	"attribute vec3 vPos;"
	"attribute vec3 vColor;"
	"uniform mat4 mat;"
	"varying vec3 color;"
	"void main(void)"
	"{"
	"color = vColor;"
	"gl_Position = mat * vec4(vPos.x, vPos.y, vPos.z, 1.0);"
	"}";

const char* color_shader_fs = "precision mediump float;"
	"varying vec3 color;"
	"void main (void)"
	"{"
	"gl_FragColor = vec4(color.x, color.y, color.z, 1.0);"
	"}";

GLuint gl_shader_create(const char* vsSource, const char* fsSource)
{
	int result = GL_FALSE;
	int infoLength = 0;

	const int vsSourceLength = strlen(vsSource);

	int vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsSource, &vsSourceLength);
	glCompileShader(vs);

	char logBuf[1024];

	glGetShaderiv(vs, GL_COMPILE_STATUS, &result);
	glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &infoLength);
	if (infoLength > 0 && result == GL_FALSE) {
		glGetShaderInfoLog(vs, infoLength, NULL, logBuf);

		ERROR_FORMAT("ERROR Message: %s", &logBuf[0]);
	}

	const int fsSourceLength = strlen(fsSource);

	int fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsSource, &fsSourceLength);
	glCompileShader(fs);

	glGetShaderiv(fs, GL_COMPILE_STATUS, &result);
	glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &infoLength);
	if (infoLength > 0 && result == GL_FALSE) {
		glGetShaderInfoLog(fs, infoLength, NULL, logBuf);

		ERROR_FORMAT("ERROR Message: %s", &logBuf[0]);
	}

	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);

	glBindAttribLocation(program, 0, "vPos");
	glBindAttribLocation(program, 1, "vColor");

	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &result);
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLength);
	if (infoLength > 0 && result == GL_FALSE) {
		glGetProgramInfoLog(program, infoLength, NULL, logBuf);

		ERROR_FORMAT("ERROR Message: %s", &logBuf[0]);
	}

	return program;
}

void gl_buffer_test(GLuint* vertexBuffer, GLuint* indexBuffer)
{
	glEnableVertexAttribArray(0); // Position
	glEnableVertexAttribArray(1); // Color

	glGenBuffers(1, vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, *vertexBuffer);

	float data[] = {
	//  X  Y  Z    R  G  B
		0, 0, 0,   1, 0, 0,
		1, 0, 0,   0, 1, 0,
		1, 1, 0,   0, 0, 1,
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 3, data, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Create the 2D IBO
	glGenBuffers(1, indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *indexBuffer);

	short index[] = { 0, 1, 2 };
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(short) * 3, index, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

static GLuint shaderColor;
static GLuint shaderColorMatrix;

static GLuint vbo;
static GLuint ibo;

void gl_setup()
{
	glClearColor(0, 0, 1, 1);

	shaderColor = gl_shader_create(color_shader_vs, color_shader_fs);
	shaderColorMatrix = glGetUniformLocation(shaderColor, "mat");
	gl_buffer_test(&vbo, &ibo);
}

void render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderColor);

	mat4 orthoMat = ortho(0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f);
	glUniformMatrix4fv(shaderColorMatrix, 1, GL_FALSE, value_ptr(orthoMat));

	// Bind stuff
	glEnableVertexAttribArray(0); // Position
	glEnableVertexAttribArray(1); // Color
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 6, (void*)(sizeof(float) * 3));

	// Draw quad
	glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);
}

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
		render();

		//Present the new frame
		if (eglSwapBuffers(eglDisplay, eglSurface) != GL_TRUE)
		{
			run = false;
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

#define CUDA_CHECK(STATUS) { cudaError_t status = (STATUS); if (status != cudaSuccess) { \
	ERROR_FORMAT("CUDA error! %s\n", cudaGetErrorName(status)); \
} } \

#define CHUNK_SIZE 16
#define CHUNK_VOXELS (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)

__global__ void generateChunkKernel(uint8_t* chunkData)
{
    int thread = threadIdx.x;

    int start = thread * (CHUNK_SIZE * CHUNK_SIZE);
    for (int x = start; x < start + (CHUNK_SIZE * CHUNK_SIZE); x++) {
        chunkData[x] = (thread+1);
    }
}

int main()
{
	if (!egl_window_create(1280, 720, L"CUDA Voxel")) 
	{
		return -1;
	}
	if (!egl_init()) 
	{
		return -1;
	}
	ShowWindow(eglWindow, SW_SHOW);

	gl_setup();
	main_loop();

  /*  CUDA_CHECK(cudaSetDevice(0));

    uint8_t* chunkDataCPU;
    uint8_t* chunkDataGPU;

    chunkDataCPU = (uint8_t*)malloc(CHUNK_VOXELS);
    memset(chunkDataCPU, 0, CHUNK_VOXELS);

    CUDA_CHECK(cudaMalloc((void**)&chunkDataGPU, CHUNK_VOXELS));
    CUDA_CHECK(cudaMemcpy(chunkDataGPU, chunkDataCPU, CHUNK_VOXELS, cudaMemcpyKind::cudaMemcpyHostToDevice));

    generateChunkKernel << <CHUNK_SIZE * CHUNK_SIZE, CHUNK_SIZE >> > (chunkDataGPU);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(chunkDataCPU, chunkDataGPU, CHUNK_VOXELS, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(chunkDataGPU));

    free(chunkDataCPU);*/

    CUDA_CHECK(cudaDeviceReset());
	egl_window_cleanup();
    return 0;
}
