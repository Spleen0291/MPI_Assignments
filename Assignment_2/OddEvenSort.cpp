#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

void odd_even_sort(vector<int>& local_data, int rank, int size, int local_n, MPI_Comm comm) {
    bool sorted = false;
    vector<int> temp(local_n);

    while (!sorted) {
        sorted = true;

        // Local Odd phase
        for (int i = 1; i < local_n; i += 2) {
            if (local_data[i] > local_data[i - 1]) {
                swap(local_data[i], local_data[i - 1]);
                sorted = false;
            }
        }
        MPI_Barrier(comm);

        // Local Even phase
        for (int i = 0; i < local_n - 1; i += 2) {
            if (local_data[i] > local_data[i + 1]) {
                swap(local_data[i], local_data[i + 1]);
                sorted = false;
            }
        }
        MPI_Barrier(comm);

        // Exchange with neighboring processes
        if (rank % 2 == 0) { // Even-ranked processes
            if (rank < size - 1) {
                int neighbor_value;
                MPI_Sendrecv(&local_data[local_n - 1], 1, MPI_INT, rank + 1, 0,
                    &neighbor_value, 1, MPI_INT, rank + 1, 0, comm, MPI_STATUS_IGNORE);
                if (neighbor_value < local_data[local_n - 1]) {
                    swap(local_data[local_n - 1], neighbor_value);
                    sorted = false;
                }
            }
        }
        else { // Odd-ranked processes
            int neighbor_value;
            MPI_Sendrecv(&local_data[0], 1, MPI_INT, rank - 1, 0,
                &neighbor_value, 1, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            if (neighbor_value > local_data[0]) {
                swap(local_data[0], neighbor_value);
                sorted = false;
            }
        }

        int global_sorted;
        MPI_Allreduce(&sorted, &global_sorted, 1, MPI_INT, MPI_LAND, comm);
        sorted = global_sorted;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10;
    vector<int> arr;
    int local_n = n / size;
    vector<int> local_data(local_n);

    srand(time(NULL) + rank);

    if (rank == 0) {
        arr.resize(n);
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 100;
        }
        cout << "Unsorted array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
    }

    MPI_Scatter(arr.data(), local_n, MPI_INT, local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    odd_even_sort(local_data, rank, size, local_n, MPI_COMM_WORLD);

    MPI_Gather(local_data.data(), local_n, MPI_INT, arr.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Sorted array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
