#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define BUCKET_SIZE 10
#define ARR_SIZE 20
#define MAX 100
#define NUM_THREADS 4 

int array[ARR_SIZE] = {29, 25, 3, 49, 9, 37, 21, 43, 1, 57, 33, 45, 17, 81, 19, 39, 59, 27, 23, 41};
int buckets[BUCKET_SIZE][ARR_SIZE];
int bucket_count[BUCKET_SIZE];

typedef struct {
    int id;
} ThreadParam;

void insertionSort(int* array, int n) {
    int i, j, key;
    for (i = 1; i < n; i++) {
        key = array[i];
        j = i - 1;
        while (j >= 0 && array[j] > key) {
            array[j + 1] = array[j];
            j = j - 1;
        }
        array[j + 1] = key;
    }
}

void* sortBuckets(void* arg) {
    int thread_id = ((ThreadParam*)arg)->id;
    int start = (thread_id * BUCKET_SIZE) / NUM_THREADS;
    int end = ((thread_id + 1) * BUCKET_SIZE) / NUM_THREADS;

    for (int i = start; i < end; i++) {
        insertionSort(buckets[i], bucket_count[i]);
    }

    pthread_exit(NULL);
}

void bucketSort() {
 
    for (int i = 0; i < ARR_SIZE; i++) {
        int index = array[i] / (MAX / BUCKET_SIZE);
        buckets[index][bucket_count[index]++] = array[i];
    }

    pthread_t threads[NUM_THREADS];
    ThreadParam params[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        params[i].id = i;
        pthread_create(&threads[i], NULL, sortBuckets, (void*)&params[i]);
    }

 
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

   
    int index = 0;
    for (int i = 0; i < BUCKET_SIZE; i++) {
        for (int j = 0; j < bucket_count[i]; j++) {
            array[index++] = buckets[i][j];
        }
    }
}

void printArray(int array[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    printf("Original array: \n");
    printArray(array, ARR_SIZE);
clock_t start_time = clock(); 
 
    bucketSort();
clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Sorted array: \n");
    printArray(array, ARR_SIZE);
  printf("Time taken for sorting: %f seconds\n", time_taken);
    return 0;
}
