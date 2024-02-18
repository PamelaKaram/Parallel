#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define ITERATION_LIMIT 255

typedef struct {
    double re;
    double im;
} ComplexNumber;

int calculateMandelbrotIterations(ComplexNumber z);
void exportToPGM(const char *filename, int buffer[IMAGE_HEIGHT][IMAGE_WIDTH]);

int main(int argc, char **argv) {
    int processId, numberOfProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcesses);

    int rowsPerProcess = IMAGE_HEIGHT / numberOfProcesses;
    int startRow = processId * rowsPerProcess;
    int endRow = (processId == numberOfProcesses - 1) ? IMAGE_HEIGHT : (processId + 1) * rowsPerProcess;

    ComplexNumber c;
    int localBuffer[rowsPerProcess][IMAGE_WIDTH];

    double executionTimeSum = 0.0;
    double averageTime = 0.0;

    for (int trial = 0; trial < 10; trial++) {
        clock_t startTime = clock();

        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < IMAGE_WIDTH; j++) {
                c.re = (j - IMAGE_WIDTH / 2.0) * 4.0 / IMAGE_WIDTH;
                c.im = (i - IMAGE_HEIGHT / 2.0) * 4.0 / IMAGE_HEIGHT;
                localBuffer[i - startRow][j] = calculateMandelbrotIterations(c);
            }
        }

        clock_t endTime = clock();
        double elapsed = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;

        MPI_Reduce(&elapsed, &averageTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (processId == 0) {
            averageTime /= numberOfProcesses;
            printf("Execution time of trial [%d]: %f seconds\n", trial, averageTime);
            executionTimeSum += averageTime;
        }
    }

    if (processId == 0) {
        printf("The average execution time of 10 trials is: %f seconds\n", executionTimeSum / 10.0);
    }

    MPI_Finalize();
    return 0;
}

int calculateMandelbrotIterations(ComplexNumber z) {
    double zReal = 0, zImaginary = 0;
    int iterations = 0;

    while (iterations < ITERATION_LIMIT && (zReal * zReal + zImaginary * zImaginary) < 4.0) {
        double tempReal = zReal * zReal - zImaginary * zImaginary + z.re;
        double tempImaginary = 2 * zReal * zImaginary + z.im;
        zReal = tempReal;
        zImaginary = tempImaginary;
        iterations++;
    }

    return iterations;
}

void exportToPGM(const char *filename, int buffer[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file\n");
        return;
    }
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            fprintf(file, "%d ", buffer[y][x]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
