// CUDA & MPI implementation of Lattice-Boltzmann with notable properties:
// 1D domain decomposition along the Y-axis
// TODO: description

#include "../../tools/config.cuh"
#include "../../tools/data_export.h"
#include "../../tools/utilities.h"
#include "initialization.cuh"
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
    constexpr uint32_t N_X_TOTAL =      60;
    constexpr uint32_t N_Y_TOTAL =      40;
    constexpr uint32_t N_STEPS =        200;
    constexpr uint64_t N_CELLS_TOTAL =  N_X_TOTAL * N_Y_TOTAL;

    // relaxation factor, rest density, max velocity, number of sine periods,
    // wavenumber (frequency), lid velocity
    constexpr FP omega = 1.5;
    constexpr FP rho_0 = 1.0;
    constexpr FP u_max = 0.1;
    constexpr FP n = 2.0;
    constexpr FP k = (FP_CONST(2.0) * FP_PI * n) / static_cast<FP>(N_Y_TOTAL);
    constexpr FP u_lid = 0.1;

    // data export settings
    uint32_t export_interval = 50;
    std::string export_name = "A";
    std::string export_num = "01";
    constexpr bool export_rho =   false;
    constexpr bool export_u_x =   true;
    constexpr bool export_u_y =   true;
    constexpr bool export_u_mag = false;

    // simulation settings
    constexpr bool shear_wave_decay =     true;
    constexpr bool lid_driven_cavity =    false;

    // =========================================================================
    // domain decomposition and MPI stuff
    // =========================================================================
    MPI_Init(&argc, &argv);

    // get the total number of processes and the rank of this process
    int SIZE, RANK;
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    // split by node
    MPI_Comm NODE_COMM;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &NODE_COMM);

    int SIZE_LOCAL, RANK_LOCAL;
    MPI_Comm_size(NODE_COMM, &SIZE_LOCAL);
    MPI_Comm_rank(NODE_COMM, &RANK_LOCAL);

    // pick GPU by local rank
    int N_GPUS_PER_NODE;
    cudaGetDeviceCount(&N_GPUS_PER_NODE);
    cudaSetDevice(RANK_LOCAL % N_GPUS_PER_NODE);

    // domain specification of this process
    // rank 0: owns rows        0   to    (Y/p) - 1
    // rank 1: owns rows    (Y/p)   to   (2Y/p) - 1
    // rank 2: owns rows   (2Y/p)   to   (3Y/p) - 1
    // rank 3: ...
    const uint32_t N_X =        N_X_TOTAL;
    const uint32_t N_Y =        N_Y_TOTAL / SIZE;
    const uint32_t Y_START =    N_Y * RANK;
    const uint32_t Y_END =      Y_START + N_Y - 1;
    const uint32_t N_CELLS =    N_X * N_Y;
    const int RANK_ABOVE =      (RANK + 1) % SIZE;
    const int RANK_BELOW =      (RANK - 1 + SIZE) % SIZE;

    if (N_Y_TOTAL % SIZE != 0)
    {
        if (RANK == 0)
        {
            SPDLOG_ERROR("Total Y must be divisible by number of processes ({} % {} != 0)",
                N_Y_TOTAL, SIZE);

            // stops all processes
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (RANK_LOCAL >= N_GPUS_PER_NODE && false)
    {
        SPDLOG_ERROR("Local rank {} wants GPU {}, but only {} GPUs found on node.",
            RANK_LOCAL, RANK_LOCAL, N_GPUS_PER_NODE);

        // stops all processes
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // =========================================================================
    // data structures and CUDA stuff
    // =========================================================================
    // host-side arrays of 9 pointers to device-side df arrays
    FP* df[9];
    FP* df_next[9];

    // for each dir i, allocate 1D array of size N_CELLS on the device
    for (uint32_t i = 0; i < 9; i++)
    {
        // (regular df arrays)
        cudaMalloc(&df[i], N_CELLS * sizeof(FP));
        cudaMalloc(&df_next[i], N_CELLS * sizeof(FP));
    }

    // domain decomposition-specific df arrays for the halo cells
    // ---------
    // | 6 2 5 |
    // | 3 0 1 |
    // | 7 4 8 |
    // ---------
    FP* df_halo_top[3]; // directions 2, 5, 6 map to indices 0, 1, 2 in the array
    FP* df_halo_bottom[3]; // directions 4, 7, 8 map to indices 0, 1, 2 in the array

    // for each select dir i, allocate 1D array of size N_CELLS on the device
    for (uint32_t i = 0; i < 3; i++)
    {
        // (halo df arrays)
        cudaMalloc(&df_halo_top[i], N_X * sizeof(FP));
        cudaMalloc(&df_halo_bottom[i], N_X * sizeof(FP));
    }

    // device-side arrays of 9 (3 for halos) pointers to device-side df arrays
    // (used as a device-side handle for the SoA data)
    FP** dvc_df;
    FP** dvc_df_next;
    cudaMalloc(&dvc_df, 9 * sizeof(FP*));
    cudaMalloc(&dvc_df_next, 9 * sizeof(FP*));
    // (halo df arrays)
    FP** dvc_df_halo_top;
    FP** dvc_df_halo_bottom;
    cudaMalloc(&dvc_df_halo_top, 3 * sizeof(FP*));
    cudaMalloc(&dvc_df_halo_bottom, 3 * sizeof(FP*));

    // copy the contents of the host-side handles to the device-side handles
    // (because apparently CUDA does not support directly passing an array of pointers)
    cudaMemcpy(dvc_df, df, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_next, df_next, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    // (halo df arrays)
    cudaMemcpy(dvc_df_halo_top, df_halo_top, 3 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_halo_bottom, df_halo_bottom, 3 * sizeof(FP*), cudaMemcpyHostToDevice);

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
    // collect buffers and other data for export context
    SimulationExportContext context;
    context.dvc_df = dvc_df;
    context.dvc_df_next = dvc_df_next;
    context.dvc_rho = dvc_rho;
    context.dvc_u_x = dvc_u_x;
    context.dvc_u_y = dvc_u_y;
    context.COMM = MPI_COMM_WORLD;
    context.N_X = N_X;
    context.N_Y = N_Y;
    context.N_X_TOTAL = N_X_TOTAL;
    context.N_Y_TOTAL = N_Y_TOTAL;
    context.Y_START = Y_START;
    context.N_CELLS = N_CELLS;
    context.RANK = RANK;

    // TODO: device and memory usage infos
    GPUInfo myInfo = GetDeviceInfos(RANK, RANK_LOCAL);
    std::vector<GPUInfo> allInfo;
    if (RANK == 0) { allInfo.resize(SIZE); }

    MPI_Gather(&myInfo, sizeof(GPUInfo), MPI_BYTE,
               allInfo.data(), sizeof(GPUInfo),
               MPI_BYTE, 0, MPI_COMM_WORLD);

    DisplayDeviceInfos(allInfo, RANK);

    if (shear_wave_decay)
    {
        Launch_ApplyInitialCondition_ShearWaveDecay_K(dvc_df, dvc_rho, dvc_u_x,
            dvc_u_y, rho_0,u_max, k, N_X, N_Y, Y_START, N_CELLS);
    }
    if (lid_driven_cavity)
    {
        Launch_ApplyInitialCondition_LidDrivenCavity_K(dvc_df, dvc_rho, dvc_u_x,
            dvc_u_y, rho_0, N_CELLS);
    }

    /* TODO
    auto inputPath = "./simulation_test_input.txt";
    if (RANK == 0 && not std::filesystem::exists(inputPath))
    {
        SPDLOG_WARN("Could not find input file {}", inputPath);
    }
    */

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

        SelectWriteBackData(step, export_interval, export_rho, export_u_x,
            export_u_y, export_u_mag, write_rho, write_u_x, write_u_y);

        // compute densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells
        Launch_FullyFusedLatticeUpdate_Push(
            dvc_df, dvc_df_next, dvc_df_halo_top, dvc_df_halo_bottom,
            dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y,
            N_X_TOTAL, N_Y_TOTAL, Y_START, Y_END, N_STEPS, N_CELLS, SIZE, RANK,
            shear_wave_decay, lid_driven_cavity, write_rho, write_u_x, write_u_y);

        // track requests for synchronization (4 per direction)
        MPI_Request requests[4 * 3];
        int req_idx = 0;

        // direction mapping for the halo arrays
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        constexpr int dir_map_halo_top[3] =     { 2, 5, 6 };
        constexpr int dir_map_halo_bottom[3] =  { 4, 7, 8 };

        /*
        if (RANK == 0 && step % 10 == 0)
        {
            FP* host_debug = new FP[N_X];

            // bottom halo in direction 4
            cudaMemcpy(host_debug, df_halo_bottom[0], N_X * sizeof(FP), cudaMemcpyDeviceToHost);
            printf("bottom halo[0]: %f %f %f %f %f %f %f %f ...\n",
                host_debug[0], host_debug[1], host_debug[2], host_debug[3],
                host_debug[4], host_debug[5], host_debug[6], host_debug[7]);

            // top ghost row in direction 4 before halo exchange
            cudaMemcpy(host_debug, df_next[4] + (N_Y - 1) * N_X, N_X * sizeof(FP), cudaMemcpyDeviceToHost);
            printf("df_next[4] top row before recv: %f %f %f %f %f %f %f %f ...\n",
                host_debug[0], host_debug[1], host_debug[2], host_debug[3],
                host_debug[4], host_debug[5], host_debug[6], host_debug[7]);

            delete[] host_debug;
        }
        */

        // TODO: send/receive halo layers into dvc_df_next while computing?
        for (uint32_t i = 0; i < 3; i++)
        {
            int dir_top = dir_map_halo_top[i];          // {2, 5, 6}
            int dir_bottom = dir_map_halo_bottom[i];    // {4, 7, 8}

            // for each of the 3 top directions, do these halo exchanges:

            // send top halo buffer to the rank above
            MPI_Isend(
                df_halo_top[i],
                N_X, FP_MPI_TYPE,
                RANK_ABOVE, dir_top,
                MPI_COMM_WORLD, &requests[req_idx++]);

            // receive the top halo from the rank above into the top row
            // (overwrite entries from (N_Y - 1) * N_X to N_Y * N_X)
            // TODO: should it not be received into the bottom row instead?
            MPI_Irecv(
                df_next[dir_top],
                N_X, FP_MPI_TYPE,
                RANK_ABOVE, dir_top,
                MPI_COMM_WORLD, &requests[req_idx++]);

            // for each of the 3 bottom directions, do these halo exchanges:

            // send bottom halo buffer to the rank below
            MPI_Isend(
                df_halo_bottom[i],
                N_X, FP_MPI_TYPE,
                RANK_BELOW, dir_bottom + 3,
                MPI_COMM_WORLD, &requests[req_idx++]);

            // receive the bottom halo from the rank below into the bottom row
            // (overwrite entries from 0 to N_X)
            // TODO: should it not be received into the top row instead?
            MPI_Irecv(
                df_next[dir_bottom] + (N_Y - 1) * N_X,
                N_X, FP_MPI_TYPE,
                RANK_BELOW, dir_bottom + 3,
                MPI_COMM_WORLD, &requests[req_idx++]);
        }

        // wait for all MPI halo exchanges to finish
        MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

        /*
        if (RANK == 0 && step % 10 == 0)
        {
            FP* host_debug = new FP[N_X];

            // top ghost row in direction 4 after halo exchange
            cudaMemcpy(host_debug, df_next[4] + (N_Y - 1) * N_X, N_X * sizeof(FP), cudaMemcpyDeviceToHost);
            printf("df_next[4] top row before recv: %f %f %f %f %f %f %f %f ...\n",
                host_debug[0], host_debug[1], host_debug[2], host_debug[3],
                host_debug[4], host_debug[5], host_debug[6], host_debug[7]);

            delete[] host_debug;
        }
        */

        std::swap(dvc_df, dvc_df_next);

        // export actual data from the arrays that have been written back to
        ExportSelectedData(context, export_name, export_num, step,
            export_interval, export_rho, export_u_x, export_u_y, export_u_mag);

        if (RANK == 0) { DisplayProgressBar(step, N_STEPS); }
    }

    if (RANK == 0)
    {
        auto end_time = std::chrono::steady_clock::now();
        // TODO: add additional metrics that are interesting for this use case
        DisplayPerformanceStats(start_time, end_time, N_X, N_Y_TOTAL, N_STEPS);
    }

    // =========================================================================
    // cleanup
    // =========================================================================
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaFree(df[i]);
        cudaFree(df_next[i]);
    }
    for (uint32_t i = 0; i < 3; i++)
    {
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
