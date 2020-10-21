#include "main.h"
#include "log.h"
#include "gl.h"

#include <Windows.h>
#include <windowsx.h>

static HDC eglHDC;
static EGLNativeWindowType eglWindow;
static EGLDisplay eglDisplay;
static EGLContext eglContext;
static EGLSurface eglSurface;

static wstring eglWindowClass = L"CudaVoxel";

static POINT screenSize;

static bool keyboardStates[256] = { false };

static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		//Destroy / close
	case WM_DESTROY:
	case WM_CLOSE:
	{
		main_exit();
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

		screenSize.x = botRight.x - topLeft.x;
		screenSize.y = botRight.y - topLeft.y;

		glViewport(0, 0, (GLsizei)(botRight.x - topLeft.x), (GLsizei)(botRight.y - topLeft.y));

		break;
	}

	// Keyboard
	case WM_KEYDOWN:
	case WM_SYSKEYDOWN:
	{
		if (wParam >= 0 && wParam < 256)
		{
			keyboardStates[wParam] = true;
		}
		break;
	}

	case WM_KEYUP:
	case WM_SYSKEYUP:
	{
		if (wParam >= 0 && wParam < 256)
		{
			keyboardStates[wParam] = false;
		}
		break;
	}
	}

	return DefWindowProcW(hWnd, message, wParam, lParam);
}

static bool egl_window_create(LONG width, LONG height, wstring title)
{
	screenSize.x = width;
	screenSize.y = height;

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

static void egl_window_cleanup()
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

static bool egl_init()
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

bool graphics_init()
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

	return true;
}

void graphics_cleanup()
{
	egl_window_cleanup();
}

bool graphics_swap()
{
	if (eglSwapBuffers(eglDisplay, eglSurface) != GL_TRUE)
	{
		main_exit();
		return false;
	}

	return true;
}

POINT graphics_size()
{
	return screenSize;
}

bool keyboard_check(int key)
{
	return keyboardStates[key];
}