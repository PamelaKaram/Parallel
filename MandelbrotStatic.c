#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define ITERATION_LIMIT 255

typedef struct {
    double re;
    double im;
} ComplexNumber;

int calculateMandelbrotIterations(ComplexNumber z) {
    double tempReal, tempImaginary, magnitudeSquared;
    int iterations = 0;
    double zReal = 0, zImaginary = 0;

    while (iterations < ITERATION_LIMIT && (zReal * zReal + zImaginary * zImaginary) < 4.0) {
        tempReal = zReal * zReal - zImaginary * zImaginary + z.re;
        tempImaginary = 2 * zReal * zImaginary + z.im;
        zReal = tempReal;
        zImaginary = tempImaginary;

        iterations++;
    }

    return iterations;
}

void exportToPGM(const char *filename, int buffer[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    FILE *file;
    file = fopen(filename, "wb");
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    fprintf(file, "255\n");
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            fprintf(file, "%d ", buffer[y][x]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int processId, numberOfProcesses;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    int rowsPerProcess = IMAGE_HEIGHT / numberOfProcesses;
    int startRow = processId * rowsPerProcess;
    int endRow = startRow + rowsPerProcess;

    ComplexNumber c;
    double executionTimeSum = 0.0;

    int localBuffer[rowsPerProcess][IMAGE_WIDTH];

    for (int trial = 0; trial < 10; ++trial) {
        clock_t startTime = clock();

        for (int row = startRow; row < endRow; ++row) {
            for (int col = 0; col < IMAGE_WIDTH; ++col) {
                c.re = (col - IMAGE_WIDTH / 2.0) * 4.0 / IMAGE_WIDTH;
                c.im = (row - IMAGE_HEIGHT / 2.0) * 4.0 / IMAGE_HEIGHT;
                
                localBuffer[row - startRow][col] = calculateMandelbrotIterations(c);
            }
        }

        clock_t endTime = clock();
        double elapsed = (double)(endTime - startTime) / CLOCKS_PER_SEC;
        double averageTime;

        MPI_Reduce(&elapsed, &averageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (processId == 0) {
            averageTime /= numberOfProcesses;
            executionTimeSum += averageTime;
            printf("Execution time of trial[%d]: %f seconds\n", trial , averageTime);
        }
    }

    if (processId == 0) {
        executionTimeSum /= 10;
        printf("Average execution time of 10 trials: %f seconds\n", executionTimeSum);
    }

    int *globalImage = NULL;
    if (processId == 0) {
        globalImage = (int *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
    }
    MPI_Gather(localBuffer, rowsPerProcess * IMAGE_WIDTH, MPI_INT, globalImage, rowsPerProcess * IMAGE_WIDTH, MPI_INT, 0, MPI_COMM_WORLD);

    if (processId == 0) {
        int finalImage[IMAGE_HEIGHT][IMAGE_WIDTH];
        for (int i = 0; i < numberOfProcesses; ++i) {
            for (int j = 0; j < rowsPerProcess; ++j) {
                for (int k = 0; k < IMAGE_WIDTH; ++k) {
                    finalImage[i * rowsPerProcess + j][k] = globalImage[i * rowsPerProcess * IMAGE_WIDTH + j * IMAGE_WIDTH + k];
                }
            }
        }
        exportToPGM("mandelbrot.pgm", finalImage);
        free(globalImage);
    }

    MPI_Finalize();

    return 0;
}
