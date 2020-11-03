#include "main.h"
#include "log.h"
#include "gl.h"

#include "chunk.h"
#include "world.h"

#include "resource.h"

#include <Windows.h>

#include <string>
using namespace std;

// Projection * View * Model

#define CAMERA_ROTATION_SPEED (3.14f)
#define CAMERA_MOVEMENT_SPEED (5.0f)

static const char* color_shader_vs = "precision highp float;"
"attribute vec3 vPos;"
"attribute vec2 vUV;"
"uniform mat4 mat;"
"varying vec2 uv;"
"void main(void)"
"{"
"uv = vUV;"
"gl_Position = mat * vec4(vPos.x, vPos.y, vPos.z, 1.0);"
"}";

static const char* color_shader_fs = "precision highp float;"
"varying vec2 uv;"
"uniform sampler2D tex;"
"void main (void)"
"{"
"gl_FragColor = texture2D(tex, uv);"
"}";

static GLuint gl_shader_create(const char* vsSource, const char* fsSource)
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

static GLuint gl_load_texture(int resourceID)
{
	HRSRC resource = FindResource(NULL, MAKEINTRESOURCEA(resourceID), __TEXT("BIN"));
	HGLOBAL resourceMemory = LoadResource(NULL, resource);

	size_t resourceSize = SizeofResource(NULL, resource);
	LPVOID resourceData = LockResource(resourceMemory);
	uint8_t* resourceDataPointer = (uint8_t*)resourceData;

	uint16_t width = *(uint16_t*)(&resourceDataPointer[0]);
	uint16_t height = *(uint16_t*)(&resourceDataPointer[2]);

	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)&resourceDataPointer[4]);

	glBindTexture(GL_TEXTURE_2D, 0);

	UnlockResource(resourceMemory);

	return textureID;
}

static GLuint shaderTexture;
static GLuint shaderTextureMatrix;
static GLuint shaderTextureSampler;

static GLuint textureBlocks;

static vec3 cameraPosition = vec3(-2, 2, -2);
static vec2 cameraRotation = vec2(0, 3.14f);

static mat4 viewProj;

static World world;

void gl_render_chunk(Chunk* chunk)
{
	int3 chunkPosition = chunk->getChunkPosition();
	mat4 world = viewProj * translate(vec3(chunkPosition.x, chunkPosition.y, chunkPosition.z) * vec3(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE));
	glUniformMatrix4fv(shaderTextureMatrix, 1, GL_FALSE, value_ptr(world));

	chunk->render();
}

void gl_setup()
{
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glFrontFace(GL_CCW);

	glEnable(GL_DEPTH_TEST);

	shaderTexture = gl_shader_create(color_shader_vs, color_shader_fs);
	shaderTextureMatrix = glGetUniformLocation(shaderTexture, "mat");
	shaderTextureSampler = glGetUniformLocation(shaderTexture, "tex");

	textureBlocks = gl_load_texture(IDR_BIN1);

	world.start();
}

void gl_cleanup()
{
	world.stop();
}

void gl_frame(float dt)
{
	// Camera
	if (keyboard_check(VK_RIGHT))
	{
		cameraRotation.y -= CAMERA_ROTATION_SPEED * dt;
	}
	if (keyboard_check(VK_LEFT))
	{
		cameraRotation.y += CAMERA_ROTATION_SPEED * dt;
	}
	if (keyboard_check(VK_UP))
	{
		cameraRotation.x += CAMERA_ROTATION_SPEED * dt;
	}
	if (keyboard_check(VK_DOWN))
	{
		cameraRotation.x -= CAMERA_ROTATION_SPEED * dt;
	}

	mat4 cameraRotationMatrix = rotate(cameraRotation.y, vec3(0, 1, 0)) * rotate(cameraRotation.x, vec3(1, 0, 0));
	vec3 cameraForward = vec3(cameraRotationMatrix * vec4(0, 0, -1, 0));
	vec3 cameraRight = vec3(cameraRotationMatrix * vec4(1, 0, 0, 0));

	float speed = (keyboard_check(VK_SHIFT) ? CAMERA_MOVEMENT_SPEED * 2.0f : CAMERA_MOVEMENT_SPEED);

	if (keyboard_check('W'))
	{
		cameraPosition += cameraForward * dt * speed;
	}
	if (keyboard_check('S'))
	{
		cameraPosition -= cameraForward * dt * speed;
	}
	if (keyboard_check('D'))
	{
		cameraPosition += cameraRight * dt * speed;
	}
	if (keyboard_check('A'))
	{
		cameraPosition -= cameraRight * dt * speed;
	}

	POINT screenSize = graphics_size();
	mat4 proj = perspectiveFov(radians(100.0f), (float)screenSize.x, (float)screenSize.y, 0.01f, 1000.0f);
	mat4 view = lookAt(cameraPosition, cameraPosition + cameraForward, vec3(0, 1, 0));

	mat4 cameraViewProj = proj * view;

	// Clear
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Setup shader
	glUseProgram(shaderTexture);

	// Bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureBlocks);
	glUniform1i(shaderTextureSampler, 0);

	// Bind VBO
	glEnableVertexAttribArray(0); // Position
	glEnableVertexAttribArray(1); // UV

	// Render world
	viewProj = cameraViewProj;

	world.render(cameraPosition);
}

void gl_create_buffer(GLuint* vertexBuffer, GLuint* indexBuffer, float* vertexData, int numVertices, uint32_t* indexData, int numIndex)
{
	glEnableVertexAttribArray(0); // Position
	glEnableVertexAttribArray(1); // UV

	glGenBuffers(1, vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, *vertexBuffer);

	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numVertices, vertexData, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Create the 2D IBO
	glGenBuffers(1, indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *indexBuffer);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * numIndex, indexData, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}