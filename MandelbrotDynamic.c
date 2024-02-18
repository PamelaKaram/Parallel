#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define ITERATION_LIMIT 255
#define WORK_REQUEST_TAG 1
#define WORK_RESPONSE_TAG 2
#define WORK_DONE_TAG 3

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

    int buffer[IMAGE_HEIGHT][IMAGE_WIDTH];
    int currentRow = 0;

    if (processId == 0) {
        // Master process
        int completedRows = 0;
        int workerStatus[numberOfProcesses];
        for (int i = 1; i < numberOfProcesses; i++) {
            workerStatus[i] = 1; // 1 indicates worker is active
        }

        while (completedRows < IMAGE_HEIGHT) {
            MPI_Status status;
            MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);

            if (currentRow < IMAGE_HEIGHT) {
                // Send a new row to work on
                MPI_Send(&currentRow, 1, MPI_INT, status.MPI_SOURCE, WORK_RESPONSE_TAG, MPI_COMM_WORLD);
                currentRow++;
            } else {
                // No more work, send termination signal
                MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, WORK_DONE_TAG, MPI_COMM_WORLD);
                workerStatus[status.MPI_SOURCE] = 0; // Mark worker as completed
            }

            // Receive completed row data
            MPI_Recv(buffer[completedRows], IMAGE_WIDTH, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            completedRows++;
        }
    } else {
        // Worker processes
        while (1) {
            MPI_Send(NULL, 0, MPI_INT, 0, WORK_REQUEST_TAG, MPI_COMM_WORLD);
            MPI_Status status;
            int rowToCompute;
            MPI_Recv(&rowToCompute, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == WORK_DONE_TAG) {
                break; // No more work
            }

            // Compute Mandelbrot set for the assigned row
            for (int j = 0; j < IMAGE_WIDTH; j++) {
                ComplexNumber c;
                c.re = (j - IMAGE_WIDTH / 2.0) * 4.0 / IMAGE_WIDTH;
                c.im = (rowToCompute - IMAGE_HEIGHT / 2.0) * 4.0 / IMAGE_HEIGHT;
                buffer[rowToCompute][j] = calculateMandelbrotIterations(c);
            }

            // Send the computed row back to the master
            MPI_Send(buffer[rowToCompute], IMAGE_WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (processId == 0) {
        exportToPGM("output.pgm", buffer);
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
