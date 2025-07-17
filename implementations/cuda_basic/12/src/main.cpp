// CUDA implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses on 1D flattened array of df values
// - fully fused kernel for density/velocity/collision/streaming operations
// - no write-back to global memory of density and velocity values

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
    // wavenumber (frequency)
    float omega = 1.2f;
    float rho_0 = 1.0f;
    float u_max = 0.1f;
    float n = 1.0f;
    float k = (2.0f * PI * n) / static_cast<float>(N_Y);

    // pointers to the device-side flattened 1D df value arrays
    float* dvc_df_flat;
    float* dvc_df_next_flat;
    cudaMalloc(&dvc_df_flat, 9 * N_CELLS * sizeof(float));
    cudaMalloc(&dvc_df_next_flat, 9 * N_CELLS * sizeof(float));

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
    //context.dvc_df = dvc_df;
    //context.dvc_df_next = dvc_df_next;
    context.dvc_rho = dvc_rho;
    context.dvc_u_x = dvc_u_x;
    context.dvc_u_y = dvc_u_y;
    context.N_X = N_X;
    context.N_Y = N_Y;

    GPUInfo myInfo = GetDeviceInfos();
    DisplayDeviceInfos(myInfo, N_X, N_Y);

    Launch_ApplyShearWaveCondition_K(dvc_df_flat, dvc_rho, dvc_u_x, dvc_u_y, rho_0,
        u_max, k, N_X, N_Y, N_CELLS);

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // update densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells
        Launch_FullyFusedOperationsComputation(
            dvc_df_flat, dvc_df_next_flat, dvc_rho, dvc_u_x, dvc_u_y, omega,
            N_X, N_Y, N_STEPS, N_CELLS);

        std::swap(dvc_df_flat, dvc_df_next_flat);

        DisplayProgressBar(step, N_STEPS);

        // export data (CAREFUL: huge file sizes)
        if (false && (step == 1 || step % 1000 == 0))
        {
            ExportSimulationData(context,
                VelocityMagnitude,
                "05",
                "A",
                step);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    DisplayPerformanceStats(start_time, end_time, N_X, N_Y, N_STEPS);

    // cleanup
    cudaFree(dvc_df_flat);
    cudaFree(dvc_df_next_flat);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    return 0;
}
