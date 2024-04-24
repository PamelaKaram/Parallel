#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS_A 900
#define COLS_A_ROWS_B 700
#define COLS_B 600


void matrixMultiply(int matA[ROWS_A][COLS_A_ROWS_B], int matB[COLS_A_ROWS_B][COLS_B], int matC[ROWS_A][COLS_B]) {
    printf("Resultant Matrix is:\n");
    for (int i = 0; i < ROWS_A; i++) {
        for (int j = 0; j < COLS_B; j++) {
            matC[i][j] = 0;
            for (int k = 0; k < COLS_A_ROWS_B; k++) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
}
int main() {
    int matrixA[ROWS_A][COLS_A_ROWS_B];
    int matrixB[COLS_A_ROWS_B][COLS_B];
    int matrixC[ROWS_A][COLS_B]; 
    clock_t start_time, end_time;
    double elapsed_time;

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
    start_time = clock();

    matrixMultiply(matrixA, matrixB, matrixC);

    end_time = clock();
    elapsed_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    return 0;
}
