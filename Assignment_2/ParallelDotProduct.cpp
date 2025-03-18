#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;  // Vector size
    std::vector<int> A, B;

    // Root process initializes vectors
    if (rank == 0) {
        A = {1, 2, 3, 4, 5, 6, 7, 8};
        B = {8, 7, 6, 5, 4, 3, 2, 1};
    }

    // Determine elements per process
    int elements_per_proc = n / size;
    std::vector<int> sub_A(elements_per_proc);
    std::vector<int> sub_B(elements_per_proc);

    // Scatter vectors among processes
    MPI_Scatter(A.data(), elements_per_proc, MPI_INT, sub_A.data(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data(), elements_per_proc, MPI_INT, sub_B.data(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local dot product
    int local_dot = 0;
    for (int i = 0; i < elements_per_proc; i++) {
        local_dot += sub_A[i] * sub_B[i];
    }

    std::cout << "Process " << rank << ": Local dot product = " << local_dot << std::endl;

    // Reduce all local dot products to compute final dot product at root
    int global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints final result
    if (rank == 0) {
        std::cout << "Final Dot Product = " << global_dot << std::endl;
    }

    MPI_Finalize();
    return 0;
}
