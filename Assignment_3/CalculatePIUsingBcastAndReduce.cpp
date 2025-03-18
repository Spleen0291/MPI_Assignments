#include <iostream>
#include <mpi.h>

static long num_steps = 100000;  // Total number of steps
double step;

int main(int argc, char* argv[]) {
    int rank, size;
    double x, sum = 0.0, pi = 0.0, partial_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Process 0 broadcasts the number of steps to all processes
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    step = 1.0 / (double)num_steps;

    // Each process computes its partial sum
    for (long i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        partial_sum += 4.0 / (1.0 + x * x);
    }

    // Reduce all partial sums to compute final pi in rank 0
    MPI_Reduce(&partial_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 computes final value of pi
    if (rank == 0) {
        pi = step * sum;
        std::cout << "Calculated PI = " << pi << std::endl;
    }

    MPI_Finalize();
    return 0;
}
