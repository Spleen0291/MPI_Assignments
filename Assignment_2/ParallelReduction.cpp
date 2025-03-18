#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8; // Total elements
    std::vector<int> data;
    
    // Root process initializes data
    if (rank == 0) {
        data = {1, 2, 3, 4, 5, 6, 7, 8};
    }

    // Determine elements per process
    int elements_per_proc = n / size;
    std::vector<int> sub_data(elements_per_proc);

    // Scatter data among processes
    MPI_Scatter(data.data(), elements_per_proc, MPI_INT, 
                sub_data.data(), elements_per_proc, MPI_INT, 
                0, MPI_COMM_WORLD);

    // Compute local sum
    int local_sum = 0;
    for (int num : sub_data) {
        local_sum += num;
    }

    std::cout << "Process " << rank << ": Local sum = " << local_sum << std::endl;

    // Perform reduction to get global sum at root process
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints final result
    if (rank == 0) {
        std::cout << "Global sum = " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
