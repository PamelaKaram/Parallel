#include <cstdio>
#include <iostream>
#include <stdio.h>

#define ROWS_A 800
#define COLS_A_ROWS_B 700
#define COLS_B 600

__global__ void matrixMultiplyKernel(int *matA, int *matB, int *matC, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; ++i) {
            sum += matA[row * colsA + i] * matB[i * colsB + col];
        }
        matC[row * colsB + col] = sum;
    }
}

int main() {
    clock_t start_time, end_time;
    double elapsed_time;
    int matrixA[ROWS_A][COLS_A_ROWS_B], matrixB[COLS_A_ROWS_B][COLS_B], matrixC[ROWS_A][COLS_B];
    int *dev_matrixA, *dev_matrixB, *dev_matrixC;
    int sizeA = ROWS_A * COLS_A_ROWS_B * sizeof(int);
    int sizeB = COLS_A_ROWS_B * COLS_B * sizeof(int);
    int sizeC = ROWS_A * COLS_B * sizeof(int);

    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_A_ROWS_B; ++j) {
            matrixA[i][j] = i + j;
        }
    }
   
    for (int i = 0; i < COLS_A_ROWS_B; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            matrixB[i][j] = i - j;
        }
    }
    cudaMalloc((void**)&dev_matrixA, sizeA);
    cudaMalloc((void**)&dev_matrixB, sizeB);
    cudaMalloc((void**)&dev_matrixC, sizeC);

 
    cudaMemcpy(dev_matrixA, matrixA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_matrixB, matrixB, sizeB, cudaMemcpyHostToDevice);
    
   
    dim3 dimBlock(COLS_B, ROWS_A);
    dim3 dimGrid(1, 1);

    
    start_time = clock();

  
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(dev_matrixA, dev_matrixB, dev_matrixC, ROWS_A, COLS_A_ROWS_B, COLS_B);


    cudaMemcpy(matrixC, dev_matrixC, sizeC, cudaMemcpyDeviceToHost);
    end_time = clock(); 
    elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    
    printf("Elapsed time: %f seconds\n", elapsed_time);

 
    cudaFree(dev_matrixA);
    cudaFree(dev_matrixB);
    cudaFree(dev_matrixC);

    return 0;
}
