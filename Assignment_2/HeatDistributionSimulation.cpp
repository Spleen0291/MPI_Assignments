#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

const int N = 4;  // Grid size
const int ITERATIONS = 500;  // Number of iterations
const double THRESHOLD = 0.0001;

typedef vector<vector<double>> Matrix;

void initialize_matrix(Matrix& grid, int rank, int size) {
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[0].size(); j++) {
            if (rank == 0 && i == 0) {
                grid[i][j] = 100.0; // Top boundary condition
            }
            else {
                grid[i][j] = 0.0;
            }
        }
    }
}

void exchange_boundaries(Matrix& grid, int rank, int size, MPI_Comm comm) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (rank > 0) {
        MPI_Send(grid[0].data(), cols, MPI_DOUBLE, rank - 1, 0, comm);
        MPI_Recv(grid[0].data(), cols, MPI_DOUBLE, rank - 1, 0, comm, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
        MPI_Recv(grid[rows - 1].data(), cols, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(grid[rows - 1].data(), cols, MPI_DOUBLE, rank + 1, 0, comm);
    }
}

void heat_distribution(Matrix& grid, int rank, int size, MPI_Comm comm) {
    int rows = grid.size();
    int cols = grid[0].size();
    Matrix new_grid = grid;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        exchange_boundaries(grid, rank, size, comm);

        double max_diff = 0.0;
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                new_grid[i][j] = 0.25 * (grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1]);
                max_diff = max(max_diff, abs(new_grid[i][j] - grid[i][j]));
            }
        }
        grid.swap(new_grid);

        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (global_max_diff < THRESHOLD) break;
    }
}

void print_grid(const Matrix& grid, int rank) {
    cout << "Rank " << rank << " final grid:\n";
    for (const auto& row : grid) {
        for (double val : row) {
            cout << val << " \t";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = N / size;
    Matrix grid(local_rows, vector<double>(N, 0.0));
    initialize_matrix(grid, rank, size);

    heat_distribution(grid, rank, size, MPI_COMM_WORLD);

    print_grid(grid, rank);

    if (rank == 0) {
        cout << "Heat distribution simulation completed." << endl;
    }

    MPI_Finalize();
    return 0;
}
