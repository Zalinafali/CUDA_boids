// Maciej Zalewski

// OpenGL
#define GLEW_STATIC
#include <GL\glew.h>
#include <GLFW\glfw3.h>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#define USE_GPU true

#define blockSize 128

#define ITERATIONS 10000

#define WIDTH 1240
#define HEIGHT 1240

#define boidsCount 10000
#define boidMaxSpeed 3.0
#define rule1Distance 75.0
#define rule1Scale 0.05
#define rule2Scale 0.1
#define rule2Distance 10.0
#define rule3Scale 0.05
#define cursorDistance 50.0
#define cursorEscapeVelocity 1.5

// calcualte distance between two boids
__device__ float magnitude(float x, float y) {
	return (float)sqrt(x*x + y * y);
}

float magnitudeCPU(float x, float y) {
	return (float)sqrt(x*x + y * y);
}

// number of blocks
int numBlocks = (boidsCount + blockSize - 1) / blockSize;

// is cursor over window
bool cursorOverWindow = false;
double cursorX;
double cursorY;
// host boid position
float* x;
float* y;
// device boid position
float* p_x;
float* p_y;
// old velocity
float* v_x1;
float* v_y1;
// new velocity
float* v_x2;
float* v_y2;

void initializeBoids() {
	cudaError_t cudaStatus;

	cudaMallocManaged(&p_x, boidsCount * sizeof(float));
	cudaMallocManaged(&p_y, boidsCount * sizeof(float));
	cudaMallocManaged(&v_x1, boidsCount * sizeof(float));
	cudaMallocManaged(&v_y1, boidsCount * sizeof(float));
	cudaMallocManaged(&v_x2, boidsCount * sizeof(float));
	cudaMallocManaged(&v_y2, boidsCount * sizeof(float));

	float* tempX = (float*)malloc(boidsCount * sizeof(float));
	float* tempY = (float*)malloc(boidsCount * sizeof(float));

	for (int i = 0; i < boidsCount; i++) {

		x[i] = ((float)rand() / RAND_MAX) * WIDTH;
		y[i] = ((float)rand() / RAND_MAX) * HEIGHT;

		tempX[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;
		tempY[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;

	}

	cudaStatus = cudaMemcpy(p_x, x, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(p_y, y, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_x1, tempX, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_y1, tempY, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_x2, tempX, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");
	cudaStatus = cudaMemcpy(v_y2, tempY, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	free(tempX);
	free(tempY);
}

void initializeBoidsCPU() {

	v_x1 = (float*)malloc(boidsCount * sizeof(float));
	v_y1 = (float*)malloc(boidsCount * sizeof(float));
	v_x2 = (float*)malloc(boidsCount * sizeof(float));
	v_y2 = (float*)malloc(boidsCount * sizeof(float));

	for (int i = 0; i < boidsCount; i++) {

		x[i] = ((float)rand() / RAND_MAX) * WIDTH;
		y[i] = ((float)rand() / RAND_MAX) * HEIGHT;

		v_x1[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;
		v_x2[i] = v_x1[i];
		v_y1[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;
		v_y2[i] = v_y1[i];

		//printf(" %f , %f\n", v_x1[i], v_y1[i]);
	}
}

void freeBoids() {
	cudaFree(p_x);
	cudaFree(p_y);
	cudaFree(v_x1);
	cudaFree(v_y1);
	cudaFree(v_x2);
	cudaFree(v_y2);
}

void freeBoidsCPU() {
	free(v_x1);
	free(v_x2);
	free(v_y1);
	free(v_y2);
}

__device__ void newVelocity(int indx, float* pos_x, float* pos_y, float* vel_x1, float* vel_y1, float* vel_x2, float* vel_y2, bool isCursorOverWindow, float curX, float curY) {
	float v[2] = { 0,0 };

	float center_of_mass[2] = { 0,0 };
	float separation[2] = { 0,0 };
	float flock_velocity[2] = { 0,0 };

	float distance = 0;
	int n = 0;

	for (int i = 0; i < boidsCount; i++) {
		if (i != indx) {
			distance = magnitude(pos_x[i] - pos_x[indx], pos_y[i] - pos_y[indx]);

			// rule 1
			if (distance < (float)rule1Distance) {
				center_of_mass[0] += pos_x[i];
				center_of_mass[1] += pos_y[i];
				n++;
			}

			// rule 2
			if (distance < (float)rule2Distance) {
				separation[0] = separation[0] - (pos_x[i] - pos_x[indx]);
				separation[1] = separation[0] - (pos_y[i] - pos_y[indx]);
			}

			// rule 3
			if (distance < (float)rule1Distance) {
				flock_velocity[0] = flock_velocity[0] + vel_x1[i];
				flock_velocity[1] = flock_velocity[1] + vel_y1[i];
			}

			if (0) {
				printf("distance: %f \n", distance);
				printf("pos i: %f, %f \n", pos_x[i], pos_y[i]);
				printf("pos indx: %f, %f \n", pos_x[indx], pos_y[indx]);
				printf("separation: %f , %f \n", separation[0], separation[1]);
				printf("flock: %f , %f \n", flock_velocity[0], flock_velocity[1]);


			}
		}
	}

	// rule 1
	if (n > 0) {
		center_of_mass[0] /= n;
		center_of_mass[1] /= n;
		v[0] = v[0] + (center_of_mass[0] - pos_x[indx]) * rule1Scale;
		v[1] = v[1] + (center_of_mass[1] - pos_y[indx]) * rule1Scale;
	}

	// rule 2
	v[0] = v[0] + separation[0] * rule2Scale;
	v[1] = v[1] + separation[1] * rule2Scale;

	// rule 3
	v[0] = v[0] + flock_velocity[0] * rule3Scale;
	v[1] = v[1] + flock_velocity[1] * rule3Scale;

	// Add to the old velocity
	v[0] = v[0] + vel_x1[indx];
	v[1] = v[1] + vel_y1[indx];

	// run away from cursor
	if (isCursorOverWindow) {
		distance = magnitude(pos_x[indx] - curX, pos_y[indx] - curY);
		if (distance < cursorDistance) {
			float normX = (pos_x[indx] - curX) / distance;
			float normY = (pos_y[indx] - curY) / distance;
			v[0] = v[0] + normX * cursorEscapeVelocity;
			v[1] = v[1] + normY * cursorEscapeVelocity;
		}
	}

	if (abs(v[0]) > boidMaxSpeed)
		vel_x2[indx] = (float)(v[0] > 0 ? boidMaxSpeed : (-1)*boidMaxSpeed);
	else
		vel_x2[indx] = v[0];

	if (abs(v[1]) > boidMaxSpeed)
		vel_y2[indx] = (float)(v[1] > 0 ? boidMaxSpeed : (-1)*boidMaxSpeed);
	else
		vel_y2[indx] = v[1];

}

void newVelocityCPU(bool isCursorOverWindow, float curX, float curY) {
	for (int indx = 0; indx < boidsCount; indx++) {
		float v[2] = { 0,0 };

		float center_of_mass[2] = { 0,0 };
		float separation[2] = { 0,0 };
		float flock_velocity[2] = { 0,0 };

		float distance = 0;
		int n = 0;

		for (int i = 0; i < boidsCount; i++) {
			if (i != indx) {
				distance = magnitudeCPU(x[i] - x[indx], y[i] - y[indx]);

				// rule 1
				if (distance < (float)rule1Distance) {
					center_of_mass[0] += x[i];
					center_of_mass[1] += y[i];
					n++;
				}

				// rule 2
				if (distance < (float)rule2Distance) {
					separation[0] = separation[0] - (x[i] - x[indx]);
					separation[1] = separation[0] - (y[i] - y[indx]);
				}

				// rule 3
				if (distance < (float)rule1Distance) {
					flock_velocity[0] = flock_velocity[0] + v_x1[i];
					flock_velocity[1] = flock_velocity[1] + v_y1[i];
				}

				if (0) {
					printf("distance: %f \n", distance);
					printf("pos i: %f, %f \n", x[i], y[i]);
					printf("pos indx: %f, %f \n", x[indx], y[indx]);
					printf("separation: %f , %f \n", separation[0], separation[1]);
					printf("flock: %f , %f \n", flock_velocity[0], flock_velocity[1]);


				}
			}
		}

		// rule 1
		if (n > 0) {
			center_of_mass[0] /= n;
			center_of_mass[1] /= n;
			v[0] = v[0] + (center_of_mass[0] - x[indx]) * rule1Scale;
			v[1] = v[1] + (center_of_mass[1] - y[indx]) * rule1Scale;
		}

		// rule 2
		v[0] = v[0] + separation[0] * rule2Scale;
		v[1] = v[1] + separation[1] * rule2Scale;

		// rule 3
		v[0] = v[0] + flock_velocity[0] * rule3Scale;
		v[1] = v[1] + flock_velocity[1] * rule3Scale;

		// Add to the old velocity
		v[0] = v[0] + v_x1[indx];
		v[1] = v[1] + v_y1[indx];

		// run away from cursor
		if (isCursorOverWindow) {
			distance = magnitudeCPU(x[indx] - curX, y[indx] - curY);
			if (distance < cursorDistance) {
				float normX = (x[indx] - curX) / distance;
				float normY = (y[indx] - curY) / distance;
				v[0] = v[0] + normX * cursorEscapeVelocity;
				v[1] = v[1] + normY * cursorEscapeVelocity;
			}
		}

		if (abs(v[0]) > boidMaxSpeed)
			v_x2[indx] = (float)(v[0] > 0 ? boidMaxSpeed : (-1)*boidMaxSpeed);
		else
			v_x2[indx] = v[0];

		if (abs(v[1]) > boidMaxSpeed)
			v_y2[indx] = (float)(v[1] > 0 ? boidMaxSpeed : (-1)*boidMaxSpeed);
		else
			v_y2[indx] = v[1];

	}
}

__global__ void updatePosition(int N, float* pos_x, float* pos_y, float* vel_x1, float* vel_y1) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N)
		return;

	pos_x[index] = pos_x[index] + vel_x1[index];
	pos_y[index] = pos_y[index] + vel_y1[index];

	if (pos_x[index] > (float)WIDTH)
		pos_x[index] = 0.0;
	if (pos_x[index] < 0.0)
		pos_x[index] = (float)WIDTH;

	if (pos_y[index] > (float)HEIGHT)
		pos_y[index] = 0.0;
	if (pos_y[index] < 0.0)
		pos_y[index] = (float)HEIGHT;

}

void updatePositionCPU() {
	for (int index = 0; index < boidsCount; index++) {
		x[index] = x[index] + v_x1[index];
		y[index] = y[index] + v_y1[index];

		if (x[index] >(float)WIDTH)
			x[index] = 0.0;
		if (x[index] < 0.0)
			x[index] = (float)WIDTH;

		if (y[index] > (float)HEIGHT)
			y[index] = 0.0;
		if (y[index] < 0.0)
			y[index] = (float)HEIGHT;
	}
}

__global__ void kernelUpdateVelocity(int N, float* pos_x, float* pos_y, float* vel_x1, float* vel_y1, float* vel_x2, float* vel_y2, bool isCursorOverWindow, float curX, float curY) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N)
		return;

	newVelocity(index, pos_x, pos_y, vel_x1, vel_y1, vel_x2, vel_y2, isCursorOverWindow, curX, curY);
}

// function for one step of the iteration with GPU
void oneStepIteration(float* pos_x, float* pos_y, float* vel_x1, float* vel_y1, float* vel_x2, float* vel_y2) {
	cudaError_t cudaStatus;

	kernelUpdateVelocity << <numBlocks, blockSize >> > ((int)boidsCount, p_x, p_y, v_x1, v_y1, v_x2, v_y2, cursorOverWindow, (float)cursorX, (float)cursorY);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	updatePosition << <numBlocks, blockSize >> > ((int)boidsCount, p_x, p_y, v_x2, v_y2);

	cudaStatus = cudaMemcpy(v_x1, v_x2, boidsCount * sizeof(float), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(v_y1, v_y2, boidsCount * sizeof(float), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
}

// function for one step iteration with CPU
void oneStepIterationCPU() {
	newVelocityCPU(cursorOverWindow, (float)cursorX, (float)cursorY);
	updatePositionCPU();
	memcpy(v_x1, v_x2, boidsCount * sizeof(float));
	memcpy(v_y1, v_y2, boidsCount * sizeof(float));
}

void prepareBoidsToDraw(GLfloat* vertices_position, float* x, float* y, float* p_x, float* p_y) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(x, p_x, boidsCount * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");
	cudaStatus = cudaMemcpy(y, p_y, boidsCount * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed!");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	for (int i = 0, j = 0; i < boidsCount; i++, j++) {
		vertices_position[j] = x[i];
		vertices_position[++j] = y[i];
	}

}

void prepareBoidsToDrawCPU(GLfloat* vertices_position, float* x, float* y) {
	for (int i = 0, j = 0; i < boidsCount; i++, j++) {
		vertices_position[j] = x[i];
		vertices_position[++j] = y[i];
	}
}

void displayFPS(GLFWwindow* window, double frameCount) {
	char fps[16];
	snprintf(fps, 16, "FPS: %f", frameCount);
	glfwSetWindowTitle(window, fps);
}

void CursorEnterCallback(GLFWwindow* window, int entered) {
	if (entered) {
		cursorOverWindow = true;
	}
	else {
		cursorOverWindow = false;
	}
}

int main(int argc, char** argv) {
	srand(time(NULL));


	x = (float*)malloc(boidsCount * sizeof(float));
	y = (float*)malloc(boidsCount * sizeof(float));

	if (USE_GPU == true)
		initializeBoids();
	else
		initializeBoidsCPU();

	// OpenGL
	if (glfwInit() != GL_TRUE) {
		std::cerr << "Fail to initialize GLFW\n";
		return -1;
	}
	else
		printf("OPNEGL OK\n");

	GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Project boids", nullptr, nullptr);		// wskaznik na okno
	if (!window)	std::cerr << "glfwCreateWindow error\n";
	glfwMakeContextCurrent(window);							// ustawienie renderingu

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Fail to initialize GLEW\n";
		return -1;
	}

	glViewport(0, 0, WIDTH, HEIGHT);
	glOrtho(0, WIDTH, 0, HEIGHT, 0, 1);

	glfwSetCursorEnterCallback(window, CursorEnterCallback);

	GLfloat* vertices_position = (GLfloat*)malloc(2 * boidsCount * sizeof(GLfloat));
	prepareBoidsToDraw(vertices_position, x, y, p_x, p_y);

	int iter = 0;
	int frameCount = 0;
	clock_t start = clock();
	double previousTime = glfwGetTime();

	while (!glfwWindowShouldClose(window)) {
		double currentTime = glfwGetTime();

		if (cursorOverWindow)
			glfwGetCursorPos(window, &cursorX, &cursorY);

		//DRAW
		glClear(GL_COLOR_BUFFER_BIT);

		glEnable(GL_POINT_SMOOTH);
		glEnableClientState(GL_VERTEX_ARRAY);
		glPointSize(2);
		glVertexPointer(2, GL_FLOAT, 0, vertices_position);
		glDrawArrays(GL_POINTS, 0, boidsCount);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisable(GL_POINT_SMOOTH);

		// move boids
		if (USE_GPU == true) {
			oneStepIteration(p_x, p_y, v_x1, v_y1, v_x2, v_y2);
			prepareBoidsToDraw(vertices_position, x, y, p_x, p_y);
		}
		else {
			oneStepIterationCPU();
			prepareBoidsToDrawCPU(vertices_position, x, y);
		}

		glfwSwapBuffers(window);

		// FPS
		frameCount++;
		if (currentTime - previousTime >= 1.0) {
			displayFPS(window, frameCount);
			frameCount = 0;
			previousTime = currentTime;
		}

		if (++iter >= ITERATIONS)
			glfwSetWindowShouldClose(window, GLFW_TRUE);

		glfwPollEvents();
	}

	clock_t end = clock();
	float loopTime = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Time: %f\n", loopTime);

	glfwTerminate();
	if (USE_GPU == true)
		freeBoids();
	else
		freeBoidsCPU();
	free(x);
	free(y);
}