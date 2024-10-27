#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void compute_primes(int max_limit, int process_rank, int total_processes) {
    // Calculate range for each process
    int range_start = 2 + process_rank * (max_limit / total_processes);
    int range_end = (process_rank == total_processes - 1) ? max_limit : range_start + (max_limit / total_processes) - 1;

    if (range_start < 2) range_start = 2;

    int range_length = range_end - range_start + 1;
    int* primes_array = malloc(range_length * sizeof(int));
    for (int i = 0; i < range_length; i++) {
        primes_array[i] = 1; // Initially assume all numbers are prime
    }

    // Implementing Sieve of Eratosthenes
    for (int num = 2; num <= sqrt(max_limit); num++) {
        int first_multiple = (range_start / num) * num;
        if (first_multiple < range_start) first_multiple += num;
        if (first_multiple == num) first_multiple += num;

        for (int j = first_multiple; j <= range_end; j += num) {
            if (j >= range_start) {
                primes_array[j - range_start] = 0;
            }
        }
    }

    // Print local prime numbers for each process
    printf("Process %d primes: ", process_rank);
    for (int i = 0; i < range_length; i++) {
        if (primes_array[i]) {
            printf("%d ", range_start + i);
        }
    }
    printf("\n");

    // Count local primes
    int local_prime_count = 0;
    for (int i = 0; i < range_length; i++) {
        if (primes_array[i]) local_prime_count++;
    }

    int* prime_counts = NULL;
    if (process_rank == 0) {
        prime_counts = malloc(total_processes * sizeof(int));
    }
    MPI_Gather(&local_prime_count, 1, MPI_INT, prime_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int overall_prime_count = 0;
    if (process_rank == 0) {
        for (int i = 0; i < total_processes; i++) {
            overall_prime_count += prime_counts[i];
        }
    }

    int* global_primes = NULL;
    int* offsets = NULL;
    if (process_rank == 0) {
        global_primes = malloc(overall_prime_count * sizeof(int));
        offsets = malloc(total_processes * sizeof(int));
        int offset = 0;
        for (int i = 0; i < total_processes; i++) {
            offsets[i] = offset;
            offset += prime_counts[i];
        }
    }

    MPI_Gatherv(primes_array, local_prime_count, MPI_INT, global_primes, prime_counts, offsets, MPI_INT, 0, MPI_COMM_WORLD);

    if (process_rank == 0) {
        printf("Total primes: %d\n", overall_prime_count);
        printf("Primes up to %d:\n", max_limit);
        for (int i = 0; i < overall_prime_count; i++) {
            printf("%d ", global_primes[i]);
        }
        printf("\n");
        free(global_primes);
        free(offsets);
    }

    free(primes_array);
    free(prime_counts);
}

int main(int argc, char** argv) {
    int rank, num_procs;
    int upper_limit;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("Enter the maximum value n: ");
        scanf("%d", &upper_limit);
    }

    MPI_Bcast(&upper_limit, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    compute_primes(upper_limit, rank, num_procs);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Execution Duration: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
