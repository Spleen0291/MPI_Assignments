#include <iostream>
#include <vector>
#include <mpi.h>

#define N 4  // Matrix size (NxN)

void print_matrix(const std::vector<int>& mat, int rows, int cols, const std::string& label) {
    std::cout << label << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = (N * N) / size;
    std::vector<int> matrix, transposed_matrix(N * N);
    std::vector<int> local_block(elements_per_proc);
    std::vector<int> transposed_block(elements_per_proc);

    // Root process initializes the matrix
    if (rank == 0) {
        matrix = { 1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12,
                   13, 14, 15, 16 };

        print_matrix(matrix, N, N, "Original Matrix");
    }

    // Scatter matrix to all processes
    MPI_Scatter(matrix.data(), elements_per_proc, MPI_INT, 
                local_block.data(), elements_per_proc, MPI_INT, 
                0, MPI_COMM_WORLD);

    // Local Transposition (swap row and column indices in local block)
    int local_size = N / size;
    for (int i = 0; i < local_size; i++) {
        for (int j = 0; j < N; j++) {
            transposed_block[j * local_size + i] = local_block[i * N + j];
        }
    }

    // Gather transposed blocks from all processes
    MPI_Gather(transposed_block.data(), elements_per_proc, MPI_INT, 
               transposed_matrix.data(), elements_per_proc, MPI_INT, 
               0, MPI_COMM_WORLD);

    // Root process prints the transposed matrix
    if (rank == 0) {
        print_matrix(transposed_matrix, N, N, "Transposed Matrix");
    }

    MPI_Finalize();
    return 0;
}
