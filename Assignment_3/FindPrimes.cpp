#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>

bool is_prime(int n) {
    if (n < 2) return false;
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    int rank, size;
    int max_value = 100;  // Set the maximum value to check for primes
    int number_to_test = 2;  // Start checking from 2
    std::vector<int> primes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Master Process
        int workers_done = 0;
        int received_value;
        MPI_Status status;

        std::cout << "Finding primes up to " << max_value << " using " << size - 1 << " workers...\n";

        while (workers_done < size - 1) {
            // Receive a message from any worker
            MPI_Recv(&received_value, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            if (received_value > 0) {
                primes.push_back(received_value);  // Store prime numbers
            }

            // Send next number to test, or 0 to signal workers to stop
            if (number_to_test <= max_value) {
                MPI_Send(&number_to_test, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                number_to_test++;
            }
            else {
                int stop_signal = 0;
                MPI_Send(&stop_signal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                workers_done++;
            }
        }

        // Print found primes
        std::cout << "Primes found: ";
        for (int p : primes) std::cout << p << " ";
        std::cout << std::endl;

    }
    else {
        // Worker Processes
        int num;
        while (true) {
            // Request a number to test
            int request = 0;
            MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Receive a number to test
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Exit if received stop signal (0)
            if (num == 0) break;

            // Check if prime
            int result = is_prime(num) ? num : -num;

            // Send result back to master
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
