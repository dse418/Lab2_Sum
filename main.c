#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>

// David Shane Elliott
// Parallel Sum

// Set length of number list and random number scale
static const long Num_To_Add = 1000000000;
static const double Scale = 10.0 / RAND_MAX;

// Serial add implementation
long add_serial(const char *numbers) {
    long sum = 0;
    for (long i = 0; i < Num_To_Add; i++) {
        sum += numbers[i];
    }
    return sum;
}

// Parallel add implementation
long add_parallel(const char *numbers) {
    // Total sum of thread sums
    long total_sum = 0;
    // Max number of threads
    int max_threads = omp_get_max_threads();
    // Calculate quantity of numbers each thread will add
    long thread_chunk_size = Num_To_Add / max_threads;
    // OpenMP parallel directive with reduction variable
#pragma omp parallel num_threads(max_threads) reduction(+: total_sum)
    {
        // Get current thread number
        int thread_num = omp_get_thread_num();
        // Calculate starting index of range of numbers to be added by thread
        long thread_chunk_start = thread_num * thread_chunk_size;
        // Calculate ending index of range of numbers to be added by thread
        long thread_chunk_end = thread_chunk_start + thread_chunk_size;
        // Initialize thread_sum
        long thread_sum = 0;
        // Iterate through range of numbers and add to thread_sum
        for (long i = thread_chunk_start; i < thread_chunk_end; i++) {
            thread_sum += numbers[i];
        }
        // Add thread_sum to total_sum
        total_sum += thread_sum;
    }
    return total_sum;
}

int main() {
    // Allocate memory and get chunk_size
    char *numbers = malloc(sizeof(long) * Num_To_Add);
    long chunk_size = Num_To_Add / omp_get_max_threads();
    // Get list of random numbers
#pragma omp parallel num_threads(omp_get_max_threads())
    {
        int p = omp_get_thread_num();
        unsigned int seed = (unsigned int) time(NULL) + (unsigned int) p;
        long chunk_start = p * chunk_size;
        long chunk_end = chunk_start + chunk_size;
        for (long i = chunk_start; i < chunk_end; i++) {
            numbers[i] = (char) (rand_r(&seed) * Scale);
        }
    }

    struct timeval start, end;

    // Get and print sequential computation time
    printf("Timing sequential...\n");
    gettimeofday(&start, NULL);
    long sum_s = add_serial(numbers);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    // Get and print parallel computation time
    printf("Timing parallel...\n");
    gettimeofday(&start, NULL);
    long sum_p = add_parallel(numbers);
    gettimeofday(&end, NULL);
    printf("Took %f seconds\n\n", end.tv_sec - start.tv_sec + (double) (end.tv_usec - start.tv_usec) / 1000000);

    // Print calculation results
    printf("Sum serial: %ld\nSum parallel: %ld", sum_s, sum_p);

    // Free allocated memory
    free(numbers);
    return 0;
}