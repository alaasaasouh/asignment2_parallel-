#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void find_primes(int limit, int proc_rank, int num_procs) {
    // Define the range for each process
    int start = 2 + proc_rank * (limit / num_procs);
    int end = (proc_rank == num_procs - 1) ? limit : start + (limit / num_procs) - 1;

    if (start < 2) start = 2;

    int range_len = end - start + 1;
    int* prime_flags = malloc(range_len * sizeof(int));
    for (int i = 0; i < range_len; i++) {
        prime_flags[i] = 1; // Assume all numbers are prime initially
    }

    // Sieve of Eratosthenes
    for (int i = 2; i <= sqrt(limit); i++) {
        int first = (start / i) * i;
        if (first < start) first += i;
        if (first == i) first += i;

        for (int j = first; j <= end; j += i) {
            if (j >= start) {
                prime_flags[j - start] = 0;
            }
        }
    }

    printf("Process %d primes: ", proc_rank);
    for (int i = 0; i < range_len; i++) {
        if (prime_flags[i]) {
            printf("%d ", start + i);
        }
    }
    printf("\n");

    int local_count = 0;
    for (int i = 0; i < range_len; i++) {
        if (prime_flags[i]) local_count++;
    }

    int* all_prime_counts = NULL;
    if (proc_rank == 0) {
        all_prime_counts = malloc(num_procs * sizeof(int));
    }
    MPI_Gather(&local_count, 1, MPI_INT, all_prime_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_count = 0;
    if (proc_rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            total_count += all_prime_counts[i];
        }
    }

    int* all_primes = NULL;
    int* displs = NULL;
    if (proc_rank == 0) {
        all_primes = malloc(total_count * sizeof(int));
        displs = malloc(num_procs * sizeof(int));
        int disp = 0;
        for (int i = 0; i < num_procs; i++) {
            displs[i] = disp;
            disp += all_prime_counts[i];
        }
    }

    MPI_Gatherv(prime_flags, local_count, MPI_INT, all_primes, all_prime_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (proc_rank == 0) {
        printf("Total prime count: %d\n", total_count);
        printf("Primes up to %d:\n", limit);
        for (int i = 0; i < total_count; i++) {
            printf("%d ", all_primes[i]);
        }
        printf("\n");
        free(all_primes);
        free(displs);
    }

    free(prime_flags);
    free(all_prime_counts);
}

int main(int argc, char** argv) {
    int rank, num_procs;
    int n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("Provide the upper bound n: ");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    find_primes(n, rank, num_procs);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Elapsed Time: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
