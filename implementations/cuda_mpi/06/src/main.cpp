// CUDA & MPI implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses of df values
// - shared memory tiling for df values
// - fully fused density/velocity/collision/streaming kernel (push)
// - lid-driven cavity with bounce-back boundary conditions
// - fp32/fp64 precision switch at compile-time for df, density, velocity values
// - int and fp versions of the directions vectors to avoid casting
// - export interval synced global write-back of density and velocity values
// - 1D domain decomposition along the Y-axis for multi-rank execution
// - additional halo arrays for push streaming and async MPI halo exchange
// - separate kernels for inner/outer cells
// - optional input.txt file for simulation parameters passed via command line
// - overlap of MPI communication and compute kernel for inner cells
// - removed branchless kernel version and made other code simplifications

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
    // logging message format: [year-month-day hour:min:sec] [type] [message]
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    // =========================================================================
    // MPI handling
    // =========================================================================
    MPI_Init(&argc, &argv);

    // get total number of processes and the rank of this process
    int RANK_SIZE, RANK;
    MPI_Comm_size(MPI_COMM_WORLD, &RANK_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    // split by node
    MPI_Comm NODE_COMM;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &NODE_COMM);

    int RANK_SIZE_LOCAL, RANK_LOCAL;
    MPI_Comm_size(NODE_COMM, &RANK_SIZE_LOCAL);
    MPI_Comm_rank(NODE_COMM, &RANK_LOCAL);

    // pick GPU by local rank
    int N_GPUS_PER_NODE = 0;
    cudaGetDeviceCount(&N_GPUS_PER_NODE);

    if (RANK_LOCAL >= N_GPUS_PER_NODE && false)
    {
        SPDLOG_ERROR("Local rank {} wants GPU {}, but only {} GPUs found on the node",
            RANK_LOCAL, RANK_LOCAL, N_GPUS_PER_NODE);

        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cudaSetDevice(RANK_LOCAL % N_GPUS_PER_NODE);

    // =========================================================================
    // parameters and input file handling
    // =========================================================================
    // default input parameters before potentially loading an input file
    SimulationParameters parameters
    {
        // scale
        .N_X_TOTAL =    1'000,
        .N_Y_TOTAL =    1'000,
        .N_STEPS =      200'000,

        // parameters
        .omega =        1.7,
        .rho_0 =        1.0,
        .u_max =        0.1,
        .u_lid =        0.1,
        .n_sin =        2.0,

        // export
        .export_interval =  50'000,
        .export_name =      "B",
        .export_num =       "06",
        .export_rho =       false,
        .export_u_x =       true,
        .export_u_y =       true,
        .export_u_mag =     false,

        // mode
        .shear_wave_decay =     false,
        .lid_driven_cavity =    true,

        // misc
        .branchless =           false
    };

    // (optional) overwrite default with simulation parameters from a file
    if (RANK == 0)
    {
        // default input path and optional overwrite via first command line arg
        std::string inputPath = (argc > 1) ? argv[1] : "<unspecified>";

        if (std::filesystem::exists(inputPath))
        {
            std::ifstream infile(inputPath);
            std::string line;
            while (std::getline(infile, line))
            {
                // skip comments and empty lines
                if (line.empty() || line[0] == '#') continue;

                std::istringstream iss(line);
                std::string key, value;
                if (!(iss >> key >> value)) continue;

                OverwriteSimulationParameters(parameters, key, value);
            }

            printf("\n");
            SPDLOG_INFO("Rank 0 loaded parameters from {}",
                inputPath);
        }
        else
        {
            printf("\n");
            SPDLOG_WARN("Rank 0 did not find input file {}, using default parameters",
                inputPath);
        }

        DisplaySimulationParameters(parameters);
    }

    MPI_Bcast(&parameters, sizeof(SimulationParameters), MPI_BYTE, 0, MPI_COMM_WORLD);

    // simulation dimension before domain decomposition
    const uint32_t N_X_TOTAL =      parameters.N_X_TOTAL;
    const uint32_t N_Y_TOTAL =      parameters.N_Y_TOTAL;
    const uint32_t N_STEPS =        parameters.N_STEPS;

    // relaxation factor, rest density, max shear wave velocity, lid velocity,
    // number of sine periods, wavenumber (frequency)
    const FP omega =    parameters.omega;
    const FP rho_0 =    parameters.rho_0;
    const FP u_max =    parameters.u_max;
    const FP u_lid =    parameters.u_lid;
    const FP n_sin =    parameters.n_sin;
    const FP w_num =    (FP_CONST(2.0) * FP_PI * n_sin) / static_cast<FP>(N_Y_TOTAL);

    // data export settings
    const uint32_t export_interval =    parameters.export_interval;
    const std::string export_name =     parameters.export_name;
    const std::string export_num =      parameters.export_num;
    const bool export_rho =             parameters.export_rho;
    const bool export_u_x =             parameters.export_u_x;
    const bool export_u_y =             parameters.export_u_y;
    const bool export_u_mag =           parameters.export_u_mag;

    // simulation mode
    const bool shear_wave_decay =       parameters.shear_wave_decay;
    const bool lid_driven_cavity =      parameters.lid_driven_cavity;

    // =========================================================================
    // domain decomposition (r = RANK_SIZE)
    // =========================================================================
    // rank 0: owns rows        0   to    (Y/r) - 1
    // rank 1: owns rows    (Y/r)   to   (2Y/r) - 1
    // rank 2: owns rows   (2Y/r)   to   (3Y/r) - 1
    // rank 3: ...
    const uint32_t N_X =            N_X_TOTAL;
    const uint32_t N_Y =            N_Y_TOTAL / RANK_SIZE;
    const uint32_t Y_START =        N_Y * RANK;
    const uint32_t N_CELLS =        N_X * N_Y;
    const uint32_t N_CELLS_INNER =  N_X * (N_Y - 2);
    const uint32_t N_CELLS_OUTER =  N_X * 2;
    const int RANK_ABOVE =          (RANK + 1) % RANK_SIZE;             // periodic
    const int RANK_BELOW =          (RANK - 1 + RANK_SIZE) % RANK_SIZE; // periodic
    const bool IS_TOP_RANK =        RANK == RANK_SIZE - 1;
    const bool IS_BOTTOM_RANK =     RANK == 0;

    if (N_Y_TOTAL % RANK_SIZE != 0)
    {
        if (RANK == 0)
        {
            SPDLOG_ERROR("Total Y must be divisible by number of processes ({} % {} != 0)",
                N_Y_TOTAL, RANK_SIZE);

            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // =========================================================================
    // data structures and CUDA stuff
    // =========================================================================
    // host-side arrays of 9 pointers to device-side df arrays
    // h_df = [ pointer -> dvc array for dir 0, ..., pointer -> dvc array for dir 8 ]
    FP* h_df[9];
    FP* h_df_new[9];

    // for each dir i, allocate 1D array of size N_CELLS on the device
    // and store pointer to it
    for (uint32_t i = 0; i < 9; i++)
    {
        // (regular df arrays)
        cudaMalloc(&h_df[i], N_CELLS * sizeof(FP));
        cudaMalloc(&h_df_new[i], N_CELLS * sizeof(FP));
    }

    // domain decomposition-specific df arrays for the halo cells
    // ---------
    // | 6 2 5 |
    // | 3 0 1 |
    // | 7 4 8 |
    // ---------
    FP* h_df_halo_top[3]; // dirs 2, 5, 6 map to indices 0, 1, 2 in the array
    FP* h_df_halo_bot[3]; // dirs 4, 7, 8 map to indices 0, 1, 2 in the array

    // for each halo-relevant dir i, allocate 1D array of size N_X on the device
    // and store pointer to it
    for (uint32_t i = 0; i < 3; i++)
    {
        // (halo df arrays)
        cudaMalloc(&h_df_halo_top[i], N_X * sizeof(FP));
        cudaMalloc(&h_df_halo_bot[i], N_X * sizeof(FP));
    }

    // device-side arrays of 9 (3 for halos) pointers to device-side df arrays
    // (same as the df[9] pointer array, but now located on the device and used
    // as a device-side handle for the SoA data used in compute kernels)
    FP** df;
    FP** df_new;
    cudaMalloc(&df, 9 * sizeof(FP*));
    cudaMalloc(&df_new, 9 * sizeof(FP*));
    // (halo df arrays)
    FP** df_halo_top;
    FP** df_halo_bot;
    cudaMalloc(&df_halo_top, 3 * sizeof(FP*));
    cudaMalloc(&df_halo_bot, 3 * sizeof(FP*));

    // copy the contents of the host-side handles to the device-side handles
    // (because CUDA does not support directly passing an array of pointers?)
    cudaMemcpy(df, h_df, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(df_new, h_df_new, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    // (halo df arrays)
    cudaMemcpy(df_halo_top, h_df_halo_top, 3 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(df_halo_bot, h_df_halo_bot, 3 * sizeof(FP*), cudaMemcpyHostToDevice);

    // pointers to the device-side density and velocity arrays
    FP* rho;
    FP* u_x;
    FP* u_y;
    cudaMalloc(&rho, N_CELLS * sizeof(FP));
    cudaMalloc(&u_x, N_CELLS * sizeof(FP));
    cudaMalloc(&u_y, N_CELLS * sizeof(FP));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // use detailed logging format for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic format
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }

    // =========================================================================
    // device info and initialization
    // =========================================================================
    // collect buffers and other data for export context
    SimulationExportContext context;
    context.dvc_df = df;
    context.dvc_df_next = df_new;
    context.dvc_rho = rho;
    context.dvc_u_x = u_x;
    context.dvc_u_y = u_y;
    context.COMM = MPI_COMM_WORLD;
    context.N_X = N_X;
    context.N_Y = N_Y;
    context.N_X_TOTAL = N_X_TOTAL;
    context.N_Y_TOTAL = N_Y_TOTAL;
    context.Y_START = Y_START;
    context.N_CELLS = N_CELLS;
    context.RANK = RANK;

    // get device and memory usage infos
    GPUInfo myInfo = GetDeviceInfos(RANK, RANK_LOCAL);
    std::vector<GPUInfo> allInfo;
    if (RANK == 0) { allInfo.resize(RANK_SIZE); }

    MPI_Gather(&myInfo, sizeof(GPUInfo), MPI_BYTE,
               allInfo.data(), sizeof(GPUInfo),
               MPI_BYTE, 0, MPI_COMM_WORLD);

    DisplayDeviceInfos(allInfo, N_X, N_Y, RANK);
    DisplayDomainDecompositionInfo(N_X, N_Y, N_X_TOTAL, N_Y_TOTAL, N_STEPS, RANK_SIZE, RANK);

    if (shear_wave_decay)
    {
        Launch_ApplyInitialCondition_ShearWaveDecay_K(
            df, rho, u_x, u_y, rho_0, u_max, w_num, N_X, N_Y, Y_START, N_CELLS);
    }
    if (lid_driven_cavity)
    {
        Launch_ApplyInitialCondition_LidDrivenCavity_K(
            df, rho, u_x, u_y, rho_0, N_CELLS);
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

        SelectWriteBackData(step, export_interval, export_rho, export_u_x,
            export_u_y, export_u_mag, write_rho, write_u_x, write_u_y);

        // track requests for synchronization (4 per direction)
        MPI_Request max_requests[4 * 3];
        int req_idx = 0;

        // dir mapping for the halo arrays
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        constexpr int dir_map_halo_top[3] = { 2, 5, 6 };
        constexpr int dir_map_halo_bot[3] = { 4, 7, 8 };

        // async MPI sends/receive halo exchange, parallel to inner cell compute
        // (no exchange between top and bottom rank for lid driven cavity,
        // BUT full exchange for shear wave decay)
        for (uint32_t i = 0; i < 3; i++)
        {
            if (step == 1) { break; }

            int dir_top = dir_map_halo_top[i]; // {2, 5, 6}
            int dir_bot = dir_map_halo_bot[i]; // {4, 7, 8}

            // for diagonal dirs, send/receive only N_X - 1 elements due to one
            // bounce-back in the corners and use offsets for the array pointers
            int offset_top = 0;
            int offset_bot = 0;
            int count = N_X;

            if (lid_driven_cavity && (i == 1 || i == 2))
            {
                count -= 1;

                // transfer of top halos in dir 5, and bottom halos in dir 7
                i == 1 ? offset_top = 1 : 0;

                // transfer of top halos in dir 6, and bottom halos in dir 8
                i == 2 ? offset_bot = 1 : 0;
            }

            // for each of the 3 top directions, do these halo exchanges:

            // send top halo buffer to the rank above
            if (not IS_TOP_RANK || shear_wave_decay)
            {
                // for a lid driven cavity, the rop rank does not do this
                MPI_Isend(
                    h_df_halo_top[i] + offset_top, count,
                    FP_MPI_TYPE, RANK_ABOVE, dir_top,
                    MPI_COMM_WORLD, &max_requests[req_idx++]);
            }

            // receive the top halo from the rank below into the own bottom row
            // (overwrite entries from 0 to N_X)
            if (not IS_BOTTOM_RANK || shear_wave_decay)
            {
                // for a lid driven cavity, the bottom rank does not do this
                MPI_Irecv(
                   h_df_new[dir_top] + offset_top, count,
                   FP_MPI_TYPE, RANK_BELOW, dir_top,
                   MPI_COMM_WORLD, &max_requests[req_idx++]);
            }

            // for each of the 3 bottom directions, do these halo exchanges:

            // send bottom halo buffer to the rank below
            if (not IS_BOTTOM_RANK || shear_wave_decay)
            {
                // for a lid driven cavity, the bottom rank does not do this
                MPI_Isend(
                    h_df_halo_bot[i] + offset_bot, count,
                    FP_MPI_TYPE, RANK_BELOW, dir_bot + 3,
                    MPI_COMM_WORLD, &max_requests[req_idx++]);
            }

            // receive the bottom halo from the rank above into the own top row
            // (overwrite entries from (N_Y - 1) * N_X to N_Y * N_X)
            if (not IS_TOP_RANK || shear_wave_decay)
            {
                // for a lid driven cavity, the rop rank does not do this
                MPI_Irecv(
                    h_df_new[dir_bot] + (N_Y - 1) * N_X + offset_bot, count,
                    FP_MPI_TYPE, RANK_ABOVE, dir_bot + 3,
                    MPI_COMM_WORLD, &max_requests[req_idx++]);
            }
        }

        // only process inner cells that don't stream to halo arrays
        // shear wave decay with pbc -> [1, ..., N_Y - 2] * N_X
        // lid driven cavity with bbbc -> [1, ..., N_Y - 1] * N_X or [0, ..., N_Y - 2] * N_X
        Launch_FullyFusedLatticeUpdate_Push_Inner(
            df, df_new, rho, u_x, u_y, omega, N_X, N_Y,
            N_X_TOTAL, N_Y_TOTAL, N_STEPS, N_CELLS_INNER, RANK, shear_wave_decay,
            lid_driven_cavity, write_rho, write_u_x, write_u_y);

        // wait for async MPI halo exchanges to finish, before outer cells can start compute
        MPI_Waitall(req_idx, max_requests, MPI_STATUSES_IGNORE);

        // only process outer cells that stream to halo arrays for 3 out of 9 directions
        // shear wave decay with pbc -> [0, N_Y - 1] * N_X
        // lid driven cavity with bbbc -> [0] * N_X or [0, N_Y - 1] * N_X
        Launch_FullyFusedLatticeUpdate_Push_Outer(
            df, df_new, df_halo_top, df_halo_bot,
            rho, u_x, u_y, omega, u_lid, N_X, N_Y, N_X_TOTAL,
            N_Y_TOTAL, Y_START, N_STEPS, N_CELLS_OUTER, RANK, shear_wave_decay,
            lid_driven_cavity, write_rho, write_u_x, write_u_y);

        // swap host pointers to the df arrays used by the MPI communication
        if (step != 1) { std::swap(h_df, h_df_new); }
        // swap device pointers to the df arrays used by the compute kernels
        std::swap(df, df_new);

        // export actual data from the arrays that have been written back to
        ExportSelectedData(context, export_name, export_num, step,
            export_interval, export_rho, export_u_x, export_u_y, export_u_mag);

        if (RANK == 0) { DisplayProgressBar(step, N_STEPS); }
    }

    if (RANK == 0)
    {
        auto end_time = std::chrono::steady_clock::now();
        // TODO: add additional metrics that are interesting for this use case
        DisplayPerformanceStats(start_time, end_time, N_X_TOTAL, N_Y_TOTAL, N_STEPS);
        SPDLOG_INFO("------------------------------------------------------\n\n\n");
    }

    // =========================================================================
    // cleanup
    // =========================================================================
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaFree(h_df[i]);
        cudaFree(h_df_new[i]);
    }
    for (uint32_t i = 0; i < 3; i++)
    {
        cudaFree(h_df_halo_top[i]);
        cudaFree(h_df_halo_bot[i]);
    }
    cudaFree(df);
    cudaFree(df_new);
    cudaFree(rho);
    cudaFree(u_x);
    cudaFree(u_y);

    MPI_Finalize();

    return 0;
}
