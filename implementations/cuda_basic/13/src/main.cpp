// CUDA implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses of df values
// - fully fused density/velocity/collision/streaming kernel (push)
// - conditional global write-back of density and velocity values
// - inlined sub-kernels for modularity (no performance impact)
// - lid-driven cavity with bounce-back boundary conditions

#include "../../tools/config.cuh"
#include "../../tools/data_export.h"
#include "../../tools/utilities.h"
#include "initialization.cuh"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [year-month-day hour:min:sec] [type] [message]
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

    constexpr float PI = 3.14159265f;

    // grid width, height, number of simulation steps, number of grid cells
    // (84 bytes per cell -> 15,000 * 10,000 cells use ~12GB of VRAM)
    uint32_t N_X =      15000;
    uint32_t N_Y =      10000;
    uint32_t N_STEPS =  1000;
    uint32_t N_CELLS =  N_X * N_Y;

    // relaxation factor, rest density, max velocity, number of sine periods,
    // wavenumber (frequency), lid velocity
    float omega = 1.2f;
    float rho_0 = 1.0f;
    float u_max = 0.1f;
    float n = 3.0f;
    float k = (2.0f * PI * n) / static_cast<float>(N_Y);
    float u_lid = 0.1f;

    // data export settings
    bool write_rho =    false;
    bool write_u_x =    false;
    bool write_u_y =    false;

    // simulation settings
    bool shear_wave_decay =     true;
    bool lid_driven_cavity =    false;

    // host-side arrays of 9 pointers to device-side df arrays
    float* df[9];
    float* df_next[9];

    // for each dir i, allocate 1D array of size N_CELLS on the device
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaMalloc(&df[i], N_CELLS * sizeof(float));
        cudaMalloc(&df_next[i], N_CELLS * sizeof(float));
    }

    // device-side arrays of 9 pointers to device-side df arrays
    // (used as a device-side handle for the SoA data)
    float** dvc_df;
    float** dvc_df_next;
    cudaMalloc(&dvc_df, 9 * sizeof(float*));
    cudaMalloc(&dvc_df_next, 9 * sizeof(float*));

    // copy the contents of the host-side handles to the device-side handles
    // (because apparently CUDA does not support directly passing an array of pointers)
    cudaMemcpy(dvc_df, df, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_next, df_next, 9 * sizeof(float*), cudaMemcpyHostToDevice);

    // pointers to the device-side density and velocity arrays
    float* dvc_rho;
    float* dvc_u_x;
    float* dvc_u_y;

    // for each array, allocate memory of size N_CELLS on the device
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(float));

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
            rho_0,u_max, k, N_X, N_Y, N_CELLS);
    }
    else if (lid_driven_cavity)
    {
        Launch_ApplyLidDrivenCavityCondition_K(dvc_df, dvc_rho, dvc_u_x,
            dvc_u_y, rho_0, N_CELLS);
    }

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // compute densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells
        Launch_FullyFusedOperationsComputation(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid,
            N_X, N_Y, N_STEPS, N_CELLS, write_rho, write_u_x, write_u_y);

        std::swap(dvc_df, dvc_df_next);

        DisplayProgressBar(step, N_STEPS);

        // export data (CAREFUL: huge file sizes)
        if (false && (step == 1 || step % 5000 == 0))
        {
            ExportSimulationData(context,
                Velocity_X,
                "13",
                "M",
                step);

            ExportSimulationData(context,
                Velocity_Y,
                "13",
                "M",
                step);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    DisplayPerformanceStats(start_time, end_time, N_X, N_Y, N_STEPS);

    // cleanup
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
