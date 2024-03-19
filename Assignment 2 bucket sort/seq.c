#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BUCKET_SIZE 10
#define ARR_SIZE 20
#define MAX 100


void insertionSort(int array[], int n) {
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


void bucketSort(int array[], int n) {
    int i, j, k;
    int buckets[BUCKET_SIZE][ARR_SIZE];
    int bucket_count[BUCKET_SIZE];

 
    for (i = 0; i < BUCKET_SIZE; i++) {
        bucket_count[i] = 0;
    }

    for (i = 0; i < n; i++) {
        int index = array[i] / (MAX / BUCKET_SIZE);
        buckets[index][bucket_count[index]++] = array[i];
    }

 
    for (i = 0, k = 0; i < BUCKET_SIZE; i++) {
        insertionSort(buckets[i], bucket_count[i]);
        for (j = 0; j < bucket_count[i]; j++) {
            array[k++] = buckets[i][j];
        }
    }
}


void printArray(int array[], int size) {
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", array[i]);
    printf("\n");
}


int main() {
    int array[ARR_SIZE] = {29, 25, 3, 49, 9, 37, 21, 43, 1, 57, 33, 45, 17, 81, 19, 39, 59, 27, 23, 41};
 
    printf("Original array: \n");
    printArray(array, ARR_SIZE);
 clock_t start_time = clock(); 
    bucketSort(array, ARR_SIZE);
clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Sorted array: \n");
    printArray(array, ARR_SIZE);
    
     printf("Time taken for sorting: %f seconds\n", time_taken);

    return 0;
}
