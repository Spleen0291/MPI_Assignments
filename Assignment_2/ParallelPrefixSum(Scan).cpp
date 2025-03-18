#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process has a single element (distributed version)
    int local_value = rank + 1;  // Example: Process 0 has 1, Process 1 has 2, etc.
    int prefix_sum = 0;

    // Perform exclusive scan (prefix sum) using MPI_Scan
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Print prefix sum from each process
    std::cout << "Process " << rank << ": Local Value = " << local_value
              << ", Prefix Sum = " << prefix_sum << std::endl;

    MPI_Finalize();
    return 0;
}
