#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int monte_carlo_pi(int num_samples) {
    int inside_circle = 0;
    for (int i = 0; i < num_samples; i++) {
        double x = static_cast<double>(rand()) / RAND_MAX;
        double y = static_cast<double>(rand()) / RAND_MAX;
        if (x * x + y * y <= 1) {
            inside_circle++;
        }
    }
    return inside_circle;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_samples = 10000000;
    int samples_per_process = total_samples / size;

    srand(time(NULL) + rank); // Ensure different seeds for each process
    int local_count = monte_carlo_pi(samples_per_process);

    int total_count;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double pi_estimate = (4.0 * total_count) / total_samples;
        cout << "Estimated value of Pi: " << pi_estimate << endl;
    }

    MPI_Finalize();
    return 0;
}
