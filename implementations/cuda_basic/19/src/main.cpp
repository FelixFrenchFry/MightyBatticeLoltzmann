// CUDA implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses of df values
// - shared memory tiling for df values
// - fully fused density/velocity/collision/streaming kernel (push)
// - inlined sub-kernels for modularity (no performance impact)
// - lid-driven cavity with bounce-back boundary conditions
// - fp32/fp64 precision switch at compile-time for df, density, velocity values
// - int and fp versions of the directions vectors to avoid casting
// - removed loop unrolling hint to reduce register pressure
// - export interval synced global write-back of density and velocity values
// - optional input.txt file for simulation parameters passed via command line

#include "../../tools/config.cuh"
#include "../../tools/data_export.h"
#include "../../tools/utilities.h"
#include "initialization.cuh"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <filesystem>
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [year-month-day hour:min:sec] [type] [message]
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    // =========================================================================
    // parameters and input file handling
    // =========================================================================
    // default input parameters before potentially loading an input file
    SimulationParameters parameters
    {
        // scale
        .N_X_TOTAL =    1000,
        .N_Y_TOTAL =    1000,
        .N_STEPS =      200000,

        // parameters
        .omega =        1.7,
        .rho_0 =        1.0,
        .u_max =        0.1,
        .u_lid =        0.1,
        .n_sin =        2.0,

        // export
        .export_interval =  50000,
        .export_name =      "A",
        .export_num =       "19",
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
        SPDLOG_INFO("Loaded parameters from {}",
            inputPath);
    }
    else
    {
        printf("\n");
        SPDLOG_WARN("Could not find input file {}, using default parameters",
            inputPath);
    }

    DisplaySimulationParameters(parameters);

    // simulation domain width, height, and number of cells before decomposition
    const uint32_t N_X =        parameters.N_X_TOTAL;
    const uint32_t N_Y =        parameters.N_Y_TOTAL;
    const uint32_t N_STEPS =    parameters.N_STEPS;
    const uint64_t N_CELLS =    N_X * N_Y;

    // relaxation factor, rest density, max shear wave velocity, lid velocity,
    // number of sine periods, wavenumber (frequency)
    const FP omega =    parameters.omega;
    const FP rho_0 =    parameters.rho_0;
    const FP u_max =    parameters.u_max;
    const FP u_lid =    parameters.u_lid;
    const FP n_sin =    parameters.n_sin;
    const FP w_num =    (FP_CONST(2.0) * FP_PI * n_sin) / static_cast<FP>(N_Y);

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

    // misc stuff
    const bool branchless =             parameters.branchless;

    // =========================================================================
    // data structures and CUDA stuff
    // =========================================================================
    // host-side arrays of 9 pointers to device-side df arrays
    FP* df[9];
    FP* df_next[9];

    // for each dir i, allocate 1D array of size N_CELLS on the device
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaMalloc(&df[i], N_CELLS * sizeof(FP));
        cudaMalloc(&df_next[i], N_CELLS * sizeof(FP));
    }

    // device-side arrays of 9 pointers to device-side df arrays
    // (used as a device-side handle for the SoA data)
    FP** dvc_df;
    FP** dvc_df_next;
    cudaMalloc(&dvc_df, 9 * sizeof(FP*));
    cudaMalloc(&dvc_df_next, 9 * sizeof(FP*));

    // copy the contents of the host-side handles to the device-side handles
    // (because apparently CUDA does not support directly passing an array of pointers)
    cudaMemcpy(dvc_df, df, 9 * sizeof(FP*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_next, df_next, 9 * sizeof(FP*), cudaMemcpyHostToDevice);

    // pointers to the device-side density and velocity arrays
    FP* dvc_rho;
    FP* dvc_u_x;
    FP* dvc_u_y;

    // for each array, allocate memory of size N_CELLS on the device
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(FP));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(FP));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(FP));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // specify detailed logging for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }

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
    context.N_X = N_X;
    context.N_Y = N_Y;

    GPUInfo myInfo = GetDeviceInfos();
    DisplayDeviceInfos(myInfo, N_X, N_Y);

    if (shear_wave_decay)
    {
        Launch_ApplyShearWaveCondition_K(dvc_df, dvc_rho, dvc_u_x, dvc_u_y,
            rho_0,u_max, w_num, N_X, N_Y, N_CELLS);
    }
    if (lid_driven_cavity)
    {
        Launch_ApplyLidDrivenCavityCondition_K(dvc_df, dvc_rho, dvc_u_x,
            dvc_u_y, rho_0, N_CELLS);
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

        // compute densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells
        Launch_FullyFusedLatticeUpdate_Push(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid,
            N_X, N_Y, N_STEPS, N_CELLS, shear_wave_decay, lid_driven_cavity,
            branchless, write_rho, write_u_x, write_u_y);

        // swap device pointers to the df arrays used by the compute kernels
        std::swap(dvc_df, dvc_df_next);

        // export actual data from the arrays that have been written back to
        ExportSelectedData(context, export_name, export_num, step,
            export_interval, export_rho, export_u_x, export_u_y, export_u_mag);

        DisplayProgressBar_Interactive(step, N_STEPS);
    }

    auto end_time = std::chrono::steady_clock::now();
    DisplayPerformanceStats(start_time, end_time, N_X, N_Y, N_STEPS);

    // =========================================================================
    // cleanup
    // =========================================================================
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaFree(df[i]);
        cudaFree(df_next[i]);
    }
    cudaFree(dvc_df);
    cudaFree(dvc_df_next);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    return 0;
}
