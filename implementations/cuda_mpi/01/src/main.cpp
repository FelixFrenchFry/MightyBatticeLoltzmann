#include <cuda_runtime.h>
#include <filesystem>
#include <mpi.h>
#include <spdlog/spdlog.h>



int main(int argc, char *argv[])
{
    // configure spdlog to display error messages like this:
    // [year-month-day hour:min:sec] [type] [message]
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    MPI_Init(&argc, &argv);

    // get the total number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get the rank of this process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // print message from each rank
    SPDLOG_INFO("Hello from process {} of {}", world_rank, world_size);
    if (world_rank == 0) { SPDLOG_INFO("I am the master process btw..."); }

    auto inputPath = "./simulation_test_input.txt";

    if (world_rank == 0 && not std::filesystem::exists(inputPath))
    {
        SPDLOG_WARN("Could not find input file {}", inputPath);
    }

    MPI_Finalize();

    return 0;
}
