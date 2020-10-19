#include "main.h"
#include "log.h"
#include "gl.h"

#include <string>
using namespace std;

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
	shaderColor = gl_shader_create(color_shader_vs, color_shader_fs);
	shaderColorMatrix = glGetUniformLocation(shaderColor, "mat");
	gl_buffer_test(&vbo, &ibo);
}

void gl_frame()
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
