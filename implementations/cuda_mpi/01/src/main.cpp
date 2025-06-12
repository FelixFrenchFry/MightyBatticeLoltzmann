// CUDA & MPI implementation of Lattice-Boltzmann with notable properties:
// 1D domain decomposition along the Y-axis
// TODO: description

#include "../../tools/config.cuh"
#include "../../tools/utilities.h"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <filesystem>
#include <mpi.h>
#include <spdlog/spdlog.h>



int main(int argc, char *argv[])
{
    // configure spdlog to display error messages like this:
    // [year-month-day hour:min:sec] [type] [message]
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    // =========================================================================
    // general parameters
    // =========================================================================
    // simulation domain width, height, and number of cells before decomposition
    constexpr uint32_t N_X_TOTAL =      150;
    constexpr uint32_t N_Y_TOTAL =      100;
    constexpr uint32_t N_STEPS =        1;
    constexpr uint64_t N_CELLS_TOTAL =  N_X_TOTAL * N_Y_TOTAL;

    // relaxation factor, rest density, max velocity, number of sine periods,
    // wavenumber (frequency), lid velocity
    constexpr FP omega = 1.2;
    constexpr FP rho_0 = 1.0;
    constexpr FP u_max = 0.1;
    constexpr FP n = 3.0;
    constexpr FP k = (FP_CONST(2.0) * FP_PI * n) / static_cast<FP>(N_Y_TOTAL);
    constexpr FP u_lid = 0.1;

    // TODO: data export settings

    // simulation settings
    constexpr bool shear_wave_decay =     true;
    constexpr bool lid_driven_cavity =    false;

    // =========================================================================
    // domain decomposition and MPI stuff
    // =========================================================================
    MPI_Init(&argc, &argv);

    // get the total number of processes and the rank of this process
    int N_PROCESSES, RANK;
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROCESSES);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    int N_GPUS_PER_NODE;
    cudaGetDeviceCount(&N_GPUS_PER_NODE);
    cudaSetDevice(RANK % N_GPUS_PER_NODE);

    // domain specification of this process
    // rank 0: owns rows        0   to    (Y/p) - 1
    // rank 1: owns rows    (Y/p)   to   (2Y/p) - 1
    // rank 2: owns rows   (2Y/p)   to   (3Y/p) - 1
    // rank 3: ...
    const uint32_t N_X =        N_X_TOTAL;
    const uint32_t N_Y =        N_Y_TOTAL / N_PROCESSES;
    const uint32_t Y_START =    N_Y * RANK;
    const uint32_t Y_END =      Y_START + N_Y - 1;
    const uint32_t N_CELLS =    N_X * N_Y;
    const int RANK_ABOVE =      (RANK + 1) % N_PROCESSES;
    const int RANK_BELOW =      (RANK - 1 + N_PROCESSES) % N_PROCESSES;

    if (N_Y_TOTAL % N_PROCESSES != 0)
    {
        if (RANK == 0)
        {
            SPDLOG_ERROR("Total Y must be divisible by number of processes ({} % {} != 0)",
                N_Y_TOTAL, N_PROCESSES);

            // stops all processes
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // =========================================================================
    // data structures and CUDA stuff
    // =========================================================================
    // host-side arrays of 9 pointers to device-side df arrays
    FP* df[9];
    FP* df_next[9];
    // domain decomposition specific df arrays for the halo cells
    FP* df_halo_top[9]; // TODO: only dirs 2, 5, and 6 need to be sent?
    FP* df_halo_bottom[9]; // TODO: only dirs 4, 7, and 8 need to be sent?

    // for each dir i, allocate 1D array of size N_CELLS on the device
    for (uint32_t i = 0; i < 9; i++)
    {
        // (regular df arrays)
        cudaMalloc(&df[i], N_CELLS * sizeof(FP));
        cudaMalloc(&df_next[i], N_CELLS * sizeof(FP));
        // (halo df arrays)
        cudaMalloc(&df_halo_top[i], N_X * sizeof(FP));
        cudaMalloc(&df_halo_bottom[i], N_X * sizeof(FP));
    }

    // device-side arrays of 9 pointers to device-side df arrays
    // (used as a device-side handle for the SoA data)
    FP** dvc_df;
    FP** dvc_df_next;
    cudaMalloc(&dvc_df, 9 * sizeof(FP*));
    cudaMalloc(&dvc_df_next, 9 * sizeof(FP*));
    // (halo df arrays)
    FP** dvc_df_halo_top;
    FP** dvc_df_halo_bottom;
    cudaMalloc(&dvc_df_halo_top, 9 * sizeof(FP*));
    cudaMalloc(&dvc_df_halo_bottom, 9 * sizeof(FP*));

    // copy the contents of the host-side handles to the device-side handles
    // (because apparently CUDA does not support directly passing an array of pointers)
    cudaMemcpy(dvc_df, df, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_next, df_next, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    // (halo df arrays)
    cudaMemcpy(dvc_df_halo_top, df_halo_top, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_halo_bottom, df_halo_bottom, 9 * sizeof(FP*), cudaMemcpyHostToDevice);

    // pointers to the device-side density and velocity arrays
    FP* dvc_rho;
    FP* dvc_u_x;
    FP* dvc_u_y;

    // for each array, allocate memory of size N_CELLS on the device
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(FP));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(FP));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(FP));

    // =========================================================================
    // device info and initialization
    // =========================================================================
    // TODO: collect buffers and other data for export context

    // TODO: device and memory usage infos

    // TODO: initial conditions

    // print info message from each rank
    std::string hello = fmt::format("Process {} of {}",
        RANK, N_PROCESSES);

    SPDLOG_INFO(hello);

    auto inputPath = "./simulation_test_input.txt";

    if (RANK == 0 && not std::filesystem::exists(inputPath))
    {
        SPDLOG_WARN("Could not find input file {}", inputPath);
    }

    if (RANK == 0)
    {
        SPDLOG_INFO("Halo cells per sub-domain: {:.2f} %",
            (2 * N_X_TOTAL * 100.0f) / (N_X_TOTAL * N_Y));
    }

    // =========================================================================
    // main simulation loop
    // =========================================================================
    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // decide which data needs global write-backs due to exports
        bool write_rho = false;
        bool write_u_x = false;
        bool write_u_y = false;

        // compute densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells
        Launch_FullyFusedLatticeUpdate_Push(
            dvc_df, dvc_df_next, dvc_df_halo_top, dvc_df_halo_bottom,
            dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y,
            N_X_TOTAL, N_Y_TOTAL, Y_START, Y_END, N_STEPS, N_CELLS, RANK,
            shear_wave_decay, lid_driven_cavity, write_rho, write_u_x, write_u_y);

        // track requests for synchronization (4 per direction)
        MPI_Request requests[4 * 9];
        int req_idx = 0;

        // TODO: send/receive halo layers into dvc_df_next while computing?
        for (uint32_t i = 0; i < 9; i++)
        {
            // this rank sends its top halo layer
            MPI_Isend(
                dvc_df_halo_top[i],     // pointer to source buffer
                N_X,                    // number of entries
                FP_MPI_TYPE,            // data type
                RANK_ABOVE,             // destination rank
                i,                      // tag
                MPI_COMM_WORLD,         // MPI communicator
                &requests[req_idx++]);  // progress tracker

            // rank above receives bottom layer
            // (one row is overwritten, from index (N_Y-1) * N_X to N_Y * N_X)
            // TODO: off by one error?
            MPI_Irecv(
                dvc_df_next[i] + (N_Y - 1) * N_X,
                N_X,
                FP_MPI_TYPE,
                RANK_ABOVE,
                i,
                MPI_COMM_WORLD,
                &requests[req_idx++]);

            // this rank sends its bottom halo layer
            MPI_Isend(
                dvc_df_halo_bottom[i],
                N_X,
                FP_MPI_TYPE,
                RANK_BELOW,
                i + 9,
                MPI_COMM_WORLD,
                &requests[req_idx++]);

            // rank below receives top layer
            // (one row is overwritten, from index 0 to N_X)
            // TODO: off by one error?
            MPI_Irecv(
                dvc_df_next[i],
                N_X,
                FP_MPI_TYPE,
                RANK_BELOW,
                i + 9,
                MPI_COMM_WORLD,
                &requests[req_idx++]);
        }

        // wait for all MPI halo exchanges to finish
        MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

        std::swap(dvc_df, dvc_df_next);

        if (RANK == 0) { DisplayProgressBar(step, N_STEPS); }
    }

    if (RANK == 0)
    {
        auto end_time = std::chrono::steady_clock::now();
        DisplayPerformanceStats(start_time, end_time, N_X, N_Y, N_STEPS);
    }

    // =========================================================================
    // cleanup
    // =========================================================================
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaFree(df[i]);
        cudaFree(df_next[i]);
        cudaFree(df_halo_top[i]);
        cudaFree(df_halo_bottom[i]);
    }
    cudaFree(dvc_df);
    cudaFree(dvc_df_next);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    MPI_Finalize();

    return 0;
}
