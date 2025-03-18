#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

const int N = 70;

void multiply_serial(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void multiply_parallel(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C, int rank, int size) {
    int rows_per_process = N / size;
    int start = rank * rows_per_process;
    int end = (rank == size - 1) ? N : start + rows_per_process;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<vector<int>> A(N, vector<int>(N, 1));
    vector<vector<int>> B(N, vector<int>(N, 1));
    vector<vector<int>> C(N, vector<int>(N, 0));

    double start_time, run_time;

    if (rank == 0) {
        start_time = omp_get_wtime();
        multiply_serial(A, B, C);
        run_time = omp_get_wtime() - start_time;
        cout << "Serial Execution Time: " << run_time << " seconds" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = omp_get_wtime();
    multiply_parallel(A, B, C, rank, size);
    run_time = omp_get_wtime() - start_time;

    if (rank == 0) {
        cout << "Parallel Execution Time: " << run_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
