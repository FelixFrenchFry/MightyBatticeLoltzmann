// CUDA & MPI implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses of df values
// - shared memory tiling for df values
// - fully fused density/velocity/collision/streaming kernel (push)
// - lid-driven cavity with bounce-back boundary conditions
// - export interval synced global write-back of density and velocity values
// - 1D domain decomposition along the Y-axis for multi-rank execution
// - additional halo arrays for push streaming and async MPI halo exchange
// - separate kernels for inner/outer cells
// - optional input.txt file for simulation parameters passed via command line
// - overlap of MPI communication and compute kernel for inner cells
// - removed branchless kernel version and made other code simplifications
// - TODO: SIMPLIFY EXPRESSIONS, DATA TYPES, COMMENTS, ... FOR PRESENTATION PURPOSES
// - TODO: THIS VERSION IS NON-FUNCTIONAL

#include "../../tools/config.cuh"
#include "../../tools/data_export.h"
#include "../../tools/utilities.h"
#include "initialization.cuh"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <filesystem>
#include <mpi.h>
#include <spdlog/spdlog.h>



int main()
{
    Kokkos::initialize();
    {
        int N = 2000;

        Kokkos::View<float*> a("a", N);
        Kokkos::View<float*> b("b", N);
        Kokkos::View<float*> c("c", N);

        // (initialize arrays...)

        Kokkos::parallel_for(N, KOKKOS_LAMBDA(int i)
        {
            c(i) = a(i) + b(i);
        });
    }
    Kokkos::finalize();
    return 0;
}

__global__ void VecAdd(float* a, float* b, float* c, int N);

int main()
{
    int N = 2000;

    float *a, *b, *c;
    cudaMalloc(&a, sizeof(float) * N);
    cudaMalloc(&b, sizeof(float) * N);
    cudaMalloc(&c, sizeof(float) * N);

    // (initialize arrays...)

    VecAdd<<<8, 256>>>(a, b, c, N);
    cudaDeviceSynchronize();

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}

__global__ void VecAdd(float* a, float* b, float* c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) { return; } 
    c[i] = a[i] + b[i];
}









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
        .export_num =       "07",
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
    const int N_X_TOTAL =      parameters.N_X_TOTAL;
    const int N_Y_TOTAL =      parameters.N_Y_TOTAL;
    const int N_STEPS =        parameters.N_STEPS;

    // relaxation factor, rest density, max shear wave velocity, lid velocity,
    // number of sine periods, wavenumber (frequency)
    const float omega =    parameters.omega;
    const float rho_0 =    parameters.rho_0;
    const float u_max =    parameters.u_max;
    const float u_lid =    parameters.u_lid;
    const float n_sin =    parameters.n_sin;
    const float w_num =    (2.0f * 3.14159265358979323846f * n_sin)
                            / static_cast<float>(N_Y_TOTAL);

    // data export settings
    const int export_interval =         parameters.export_interval;
    const std::string export_name =     parameters.export_name;
    const std::string export_num =      parameters.export_num;
    const bool export_rho =             parameters.export_rho;
    const bool export_u_x =             parameters.export_u_x;
    const bool export_u_y =             parameters.export_u_y;
    const bool export_u_mag =           parameters.export_u_mag;

    // simulation mode
    const bool is_SWD =     parameters.shear_wave_decay;
    const bool is_LDC =     parameters.lid_driven_cavity;

    // =========================================================================
    // domain decomposition (r = RANK_SIZE)
    // =========================================================================
    // rank 0: owns rows        0   to    (Y/r) - 1
    // rank 1: owns rows    (Y/r)   to   (2Y/r) - 1
    // rank 2: owns rows   (2Y/r)   to   (3Y/r) - 1
    // rank 3: ...
    const int N_X =                 N_X_TOTAL;
    const int N_Y =                 N_Y_TOTAL / RANK_SIZE;
    const int Y_START =             N_Y * RANK;
    const int N_CELLS =             N_X * N_Y;
    const int N_CELLS_INNER =       N_X * (N_Y - 2);
    const int N_CELLS_OUTER =       N_X * 2;
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

    /*
    // =========================================================================
    // data structures and CUDA stuff
    // =========================================================================
    // host-side arrays of 9 pointers to device-side df arrays
    // h_df = [ pointer -> dvc array for dir 0, ..., pointer -> dvc array for dir 8 ]
    float* df[9];
    float* df_new[9];

    // for each dir i, allocate 1D array of size N_CELLS on the device
    // and store pointer to it
    for (uint32_t i = 0; i < 9; i++)
    {
        // (regular df arrays)
        cudaMalloc(&df[i], N_CELLS * sizeof(float));
        cudaMalloc(&df_new[i], N_CELLS * sizeof(float));
    }

    // domain decomposition-specific df arrays for the halo cells
    // ---------
    // | 6 2 5 |
    // | 3 0 1 |
    // | 7 4 8 |
    // ---------
    // TODO: extend this to 9 arrays, but use placeholders for irrelevant ones
    float* df_halo_top[9]; // indices 2, 5, 6 correspond to the relevant dirs
    float* df_halo_bot[9]; // indices 4, 7, 8 correspond to the relevant dirs

    // for each halo-relevant dir i, allocate 1D array of size N_X on the device
    // and store pointer to it
    for (uint32_t i = 0; i < 9; i++)
    {
        // (halo df arrays)
        cudaMalloc(&df_halo_top[i], N_X * sizeof(float));
        cudaMalloc(&df_halo_bot[i], N_X * sizeof(float));
    }

    // device-side arrays of 9 pointers to device-side df arrays
    // (same as the df[9] pointer array, but now located on the device and used
    // as a device-side handle for the SoA data used in compute kernels)
    float** dvc_df;
    float** dvc_df_new;
    cudaMalloc(&dvc_df, 9 * sizeof(float*));
    cudaMalloc(&dvc_df_new, 9 * sizeof(float*));
    // (halo df arrays)
    float** dvc_df_halo_top;
    float** dvc_df_halo_bot;
    cudaMalloc(&dvc_df_halo_top, 9 * sizeof(float*));
    cudaMalloc(&dvc_df_halo_bot, 9 * sizeof(float*));

    // copy the contents of the host-side handles to the device-side handles
    // (because CUDA does not support directly passing an array of pointers?)
    cudaMemcpy(dvc_df, df, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_new, df_new, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    // (halo df arrays)
    cudaMemcpy(dvc_df_halo_top, df_halo_top, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_halo_bot, df_halo_bot, 9 * sizeof(float*), cudaMemcpyHostToDevice);

    // pointers to the device-side density and velocity arrays
    float* dvc_rho;
    float* dvc_u_x;
    float* dvc_u_y;
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(float));
    */

    // TODO: DATA STRUCTURES SECTION FOR PRESENTATION SCREENSHOTS
    // TODO: NON-FUNCTIONAL

    // array of pointers to the 9 df arrays in GPU memory
    float* df[9];
    for (int i = 0; i < 9; i++)
    {
        cudaMalloc(&df[i], sizeof(float) * N_C);
    }

    // mirrored on the GPU, pointers copied into it
    float** dvc_df;
    cudaMalloc(&dvc_df, sizeof(float*) * 9);
    cudaMemcpy(dvc_df, df, sizeof(float*) * 9,
               cudaMemcpyHostToDevice);

    // "_new" versions for writing
    float* df_new[9];
    ...

    float** dvc_df_new;
    ...


    // pointers to density and velocity arrays on the GPU
    float* dvc_rho;
    float* dvc_u_x;
    float* dvc_u_y;
    cudaMalloc(&dvc_rho, sizeof(float) * N_C);
    cudaMalloc(&dvc_u_x, sizeof(float) * N_C);
    cudaMalloc(&dvc_u_y, sizeof(float) * N_C);

    // =========================================================================
    // data structures and CUDA stuff
    // =========================================================================
    // host-side arrays of 9 pointers to device-side df arrays
    // h_df = [ pointer -> dvc array for dir 0, ..., pointer -> dvc array for dir 8 ]
    float* df[9];
    float* df_new[9];

    // for each dir i, allocate 1D array of size N_CELLS on the device
    // and store pointer to it
    for (uint32_t i = 0; i < 9; i++)
    {
        // (regular df arrays)
        cudaMalloc(&df[i], N_CELLS * sizeof(float));
        cudaMalloc(&df_new[i], N_CELLS * sizeof(float));
    }

    // domain decomposition-specific df arrays for the halo cells
    // ---------
    // | 6 2 5 |
    // | 3 0 1 |
    // | 7 4 8 |
    // ---------
    // TODO: extend this to 9 arrays, but use placeholders for irrelevant ones
    float* df_halo_top[9]; // indices 2, 5, 6 correspond to the relevant dirs
    float* df_halo_bot[9]; // indices 4, 7, 8 correspond to the relevant dirs

    // for each halo-relevant dir i, allocate 1D array of size N_X on the device
    // and store pointer to it
    for (uint32_t i = 0; i < 9; i++)
    {
        // (halo df arrays)
        cudaMalloc(&df_halo_top[i], N_X * sizeof(float));
        cudaMalloc(&df_halo_bot[i], N_X * sizeof(float));
    }

    // device-side arrays of 9 pointers to device-side df arrays
    // (same as the df[9] pointer array, but now located on the device and used
    // as a device-side handle for the SoA data used in compute kernels)
    float** dvc_df;
    float** dvc_df_new;
    cudaMalloc(&dvc_df, 9 * sizeof(float*));
    cudaMalloc(&dvc_df_new, 9 * sizeof(float*));
    // (halo df arrays)
    float** dvc_df_halo_top;
    float** dvc_df_halo_bot;
    cudaMalloc(&dvc_df_halo_top, 9 * sizeof(float*));
    cudaMalloc(&dvc_df_halo_bot, 9 * sizeof(float*));

    // copy the contents of the host-side handles to the device-side handles
    // (because CUDA does not support directly passing an array of pointers?)
    cudaMemcpy(dvc_df, df, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_new, df_new, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    // (halo df arrays)
    cudaMemcpy(dvc_df_halo_top, df_halo_top, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_halo_bot, df_halo_bot, 9 * sizeof(float*), cudaMemcpyHostToDevice);

    // pointers to the device-side density and velocity arrays
    float* dvc_rho;
    float* dvc_u_x;
    float* dvc_u_y;
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(float));

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
    context.dvc_df = dvc_df;
    context.dvc_df_next = dvc_df_new;
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

    // get device and memory usage infos
    GPUInfo myInfo = GetDeviceInfos(RANK, RANK_LOCAL);
    std::vector<GPUInfo> allInfo;
    if (RANK == 0) { allInfo.resize(RANK_SIZE); }

    MPI_Gather(&myInfo, sizeof(GPUInfo), MPI_BYTE,
               allInfo.data(), sizeof(GPUInfo),
               MPI_BYTE, 0, MPI_COMM_WORLD);

    DisplayDeviceInfos(allInfo, N_X, N_Y, RANK);
    DisplayDomainDecompositionInfo(N_X, N_Y, N_X_TOTAL, N_Y_TOTAL, N_STEPS, RANK_SIZE, RANK);

    if (is_SWD)
    {
        Launch_ApplyInitialCondition_ShearWaveDecay_K(dvc_df, dvc_rho,
            dvc_u_x, dvc_u_y, rho_0, u_max, w_num, N_X, N_Y, Y_START, N_CELLS);
    }
    if (is_LDC)
    {
        Launch_ApplyInitialCondition_LidDrivenCavity_K(
            dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0, N_CELLS);
    }

    // TODO: SIMULATION LOOP FOR PRESENTATION PURPOSES
    // TODO: NON-FUNCTIONAL

    int N_BLOCKS_INNER = (N_CELLS_INNER + 255) / 256;
    int N_BLOCKS_OUTER = (N_CELLS_OUTER + 255) / 256;

    for (int step = 0; step < N_STEPS; step++)
    {
        // exchange halo cells asynchronously
        if (step > 0) { AsyncMPICommunication(...); }

        Kernel_InnerCells<<<N_BLOCKS_INNER, 256>>>(...);

        // wait for halo exchange to finish
        MPI_Waitall(...);

        Kernel_OuterCells<<<N_BLOCKS_OUTER, 256>>>(...);

        cudaDeviceSynchronize();

        // swap pointers for old/new DF data
        std::swap(dvc_df, dvc_df_new);
        std::swap(df, df_new);
    }

    // =========================================================================
    // main simulation loop
    // =========================================================================
    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // decide which data needs global write-backs due to exports
        bool save_rho = false;
        bool save_u_x = false;
        bool save_u_y = false;

        SelectWriteBackData(step, export_interval, export_rho, export_u_x,
            export_u_y, export_u_mag, save_rho, save_u_x, save_u_y);

        // track requests for synchronization (4 per direction)
        MPI_Request max_requests[4 * 3];
        int req_idx = 0;

        // dir mapping for the halo arrays
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------

        // TODO: USE THIS FOR PRESENTATION SCREENSHOTS ONLY
        // TODO: THIS IS NON-FUNCTIONAL
        if (not IS_TOP_RANK)
        {
            // send top halo to the rank above
            MPI_Isend(df_halo_top[2], N_X,
                      MPI_FLOAT, RANK_ABOVE, 2,
                      MPI_COMM_WORLD, &max_requests[req_idx++]);

            // receive bottom halo from the rank above, into the own top row
            MPI_Irecv(df[4] + (N_Y - 1) * N_X, N_X,
                      MPI_FLOAT, RANK_ABOVE, 4,
                      MPI_COMM_WORLD, &max_requests[req_idx++]);
        }
        if (not IS_BOTTOM_RANK)
        {
            // send bottom halo to the rank below
            MPI_Isend(df_halo_bot[4], N_X,
                      MPI_FLOAT, RANK_BELOW, 4,
                      MPI_COMM_WORLD, &max_requests[req_idx++]);

            // receive the top halo from the rank below, into the own bottom row
            MPI_Irecv(df[2], N_X,
                      MPI_FLOAT, RANK_BELOW, 2,
                      MPI_COMM_WORLD, &max_requests[req_idx++]);
        }

        constexpr int dir_map_halo_top[3] = { 2, 5, 6 };
        constexpr int dir_map_halo_bot[3] = { 4, 7, 8 };

        // async MPI sends/receive halo exchange, parallel to inner cell compute
        // (no exchange between top and bottom rank for lid driven cavity,
        // BUT full exchange for shear wave decay)
        for (uint32_t i = 0; i < 3; i++)
        {
            // IMPORTANT TO NOT OVERWRITE THE INITIALIZED VALUES WITH TRASH
            if (step == 1) { break; }

            int dir_top = dir_map_halo_top[i]; // {2, 5, 6}
            int dir_bot = dir_map_halo_bot[i]; // {4, 7, 8}

            // for diagonal dirs, send/receive only N_X - 1 elements due to one
            // bounce-back in the corners and use offsets for the array pointers
            int offset_top = 0;
            int offset_bot = 0;
            int count = N_X;

            if (is_LDC && (i == 1 || i == 2))
            {
                count -= 1;

                // transfer of top halos in dir 5, and bottom halos in dir 7
                i == 1 ? offset_top = 1 : 0;

                // transfer of top halos in dir 6, and bottom halos in dir 8
                i == 2 ? offset_bot = 1 : 0;
            }

            // for each of the 3 top directions, do these halo exchanges:

            // send top halo buffer to the rank above
            if (not IS_TOP_RANK || is_SWD)
            {
                // for a lid driven cavity, the rop rank does not do this
                MPI_Isend(
                    df_halo_top[i] + offset_top, count,
                    MPI_FLOAT, RANK_ABOVE, dir_top,
                    MPI_COMM_WORLD, &max_requests[req_idx++]);
            }

            // receive the top halo from the rank below into the own bottom row
            // (overwrite entries from 0 to N_X)
            if (not IS_BOTTOM_RANK || is_SWD)
            {
                // for a lid driven cavity, the bottom rank does not do this
                MPI_Irecv(
                   df[dir_top] + offset_top, count,
                   MPI_FLOAT, RANK_BELOW, dir_top,
                   MPI_COMM_WORLD, &max_requests[req_idx++]);
            }

            // for each of the 3 bottom directions, do these halo exchanges:

            // send bottom halo buffer to the rank below
            if (not IS_BOTTOM_RANK || is_SWD)
            {
                // for a lid driven cavity, the bottom rank does not do this
                MPI_Isend(
                    df_halo_bot[i] + offset_bot, count,
                    MPI_FLOAT, RANK_BELOW, dir_bot + 3,
                    MPI_COMM_WORLD, &max_requests[req_idx++]);
            }

            // receive the bottom halo from the rank above into the own top row
            // (overwrite entries from (N_Y - 1) * N_X to N_Y * N_X)
            if (not IS_TOP_RANK || is_SWD)
            {
                // for a lid driven cavity, the rop rank does not do this
                MPI_Irecv(
                    df[dir_bot] + (N_Y - 1) * N_X + offset_bot, count,
                    MPI_FLOAT, RANK_ABOVE, dir_bot + 3,
                    MPI_COMM_WORLD, &max_requests[req_idx++]);
            }
        }

        // only process inner cells that don't stream to halo arrays
        // LDC with bbbc -> [1, ..., N_Y - 1] * N_X or [0, ..., N_Y - 2] * N_X
        Launch_FullyFusedLatticeUpdate_Push_Inner(
            dvc_df, dvc_df_new, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
            N_X_TOTAL, N_Y_TOTAL, N_STEPS, N_CELLS_INNER, RANK, is_SWD, is_LDC,
            save_rho, save_u_x, save_u_y);

        // TODO: do this here instead of the launcher
        cudaDeviceSynchronize();

        // wait for async MPI halo exchanges to finish, before outer cells can start compute
        MPI_Waitall(req_idx, max_requests, MPI_STATUSES_IGNORE);

        // only process outer cells that stream to halo arrays
        // LDC with bbbc -> [0] * N_X or [0, N_Y - 1] * N_X
        Launch_FullyFusedLatticeUpdate_Push_Outer(
            dvc_df, dvc_df_new, dvc_df_halo_top, dvc_df_halo_bot, dvc_rho,
            dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y, N_X_TOTAL, N_Y_TOTAL,
            Y_START, N_STEPS, N_CELLS_OUTER, RANK, is_SWD, is_LDC,
            save_rho, save_u_x, save_u_y);

        // swap host pointers to the df arrays used by the MPI communication
        std::swap(df, df_new);
        // swap device pointers to the df arrays used by the compute kernels
        std::swap(dvc_df, dvc_df_new);

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
        cudaFree(df[i]);
        cudaFree(df_new[i]);
    }
    for (uint32_t i = 0; i < 3; i++)
    {
        cudaFree(df_halo_top[i]);
        cudaFree(df_halo_bot[i]);
    }
    cudaFree(dvc_df);
    cudaFree(dvc_df_new);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    MPI_Finalize();

    return 0;
}
