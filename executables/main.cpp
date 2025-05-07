#include "hello.h"
#include <iostream>
#include <filesystem>

#ifdef WITH_MPI
#include <mpi.h>
#endif


int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    // Below is some MPI code, try compiling with `cmake -DWITH_MPI=ON ..`
#ifdef WITH_MPI
    MPI_Init(&argc, &argv);

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    std::cout << "Hello I am rank " << rank << " of " << size << "\n";

    if (rank == 0)
      hello_world();

    auto input_path = "./simulation_test_input.txt";

    if (not std::filesystem::exists(input_path))
      std::cerr << "warning: could not find input file " << input_path << "\n";

#ifdef WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}
