#include <iostream>
#include <vector>
#include <mpi.h>

#define N (1 << 16)  // 2^16 elements

void daxpy_serial(double a, std::vector<double>& X, const std::vector<double>& Y) {
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void daxpy_parallel(double a, std::vector<double>& X, const std::vector<double>& Y, int rank, int size) {
    int chunk_size = N / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? N : start + chunk_size;

    for (int i = start; i < end; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 2.5;
    std::vector<double> X(N, 1.0);  // Initialize X with 1.0
    std::vector<double> Y(N, 2.0);  // Initialize Y with 2.0

    std::vector<double> recv_X;  // Separate receive buffer

    // Measure serial execution time on rank 0
    double serial_start, serial_end, parallel_start, parallel_end;
    if (rank == 0) {
        serial_start = MPI_Wtime();
        daxpy_serial(a, X, Y);
        serial_end = MPI_Wtime();
        recv_X.resize(N);  // Allocate memory for final result
    }

    // Broadcast the vector X after modification in serial execution
    MPI_Bcast(X.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Measure parallel execution time
    parallel_start = MPI_Wtime();
    daxpy_parallel(a, X, Y, rank, size);
    parallel_end = MPI_Wtime();

    // Gather results from all processes into recv_X (not X!)
    MPI_Gather(X.data() + (rank * (N / size)), N / size, MPI_DOUBLE, 
               recv_X.data(), N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute speedup
    if (rank == 0) {
        double serial_time = serial_end - serial_start;
        double parallel_time = parallel_end - parallel_start;
        double speedup = serial_time / parallel_time;

        std::cout << "Serial Execution Time: " << serial_time << " seconds\n";
        std::cout << "Parallel Execution Time: " << parallel_time << " seconds\n";
        std::cout << "Speedup: " << speedup << "x\n";
    }

    MPI_Finalize();
    return 0;
}
