#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

void performMatrixMultiplication(int* matA, int* matB, int* matC, int numRowsA, int numColsA, int numColsB);

int main() {
    int numRowsA, numColsA, numColsB;

    printf("Enter the dimensions of matrix A (numRowsA): ");
    scanf("%d", &numRowsA);
    printf("Enter the dimensions of matrix B (numColsA x numColsB): ");
    scanf("%d %d", &numColsA, &numColsB);

    int* matA, *matB, *matC;

    matA = (int*)malloc(numRowsA * numColsA * sizeof(int));
    matB = (int*)malloc(numColsA * numColsB * sizeof(int));
    matC = (int*)malloc(numRowsA * numColsB * sizeof(int));

    srand(time(NULL)); 
    for (int i = 0; i < numRowsA * numColsA; ++i) {
        matA[i] = rand() % 10;
    }
    for (int i = 0; i < numColsA * numColsB; ++i) {
        matB[i] = rand() % 10;
    }

    double start = omp_get_wtime();
    performMatrixMultiplication(matA, matB, matC, numRowsA, numColsA, numColsB);
    double end = omp_get_wtime();
    printf("Execution Time: %f seconds\n", end - start);

    free(matA);
    free(matB);
    free(matC);

    return 0;
}

void performMatrixMultiplication(int* matA, int* matB, int* matC, int numRowsA, int numColsA, int numColsB) {
    #pragma acc parallel loop collapse(2) present(matA[:numRowsA*numColsA], matB[:numColsA*numColsB], matC[:numRowsA*numColsB])
    for (int row = 0; row < numRowsA; ++row) {
        for (int col = 0; col < numColsB; ++col) {
            int sum = 0;
            for (int k = 0; k < numColsA; ++k) {
                sum += matA[row * numColsA + k] * matB[k * numColsB + col];
            }
            matC[row * numColsB + col] = sum;
        }
    }
}
