// CUDA implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses of df values
// - shared memory tiling for df values
// - fully fused kernel for density/velocity/collision/streaming operations
// - explicitly unrolled loops, combined with manually optimized computations

#include "../../tools_fp32/data_export.h"
#include "../../tools_fp32/utilities.h"
#include "config.cuh"
#include "fullyfused.cuh"
#include "initialization.cuh"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // ----- INITIALIZATION OF MISC STUFF -----

    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    constexpr float PI = 3.14159265f;

    // TODO: check important hardware properties
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    SPDLOG_INFO("GPU: {} (CC: {})", props.name, props.major * 10 + props.minor);

    // ----- INITIALIZATION OF PARAMETERS -----

    // grid width, height, number of simulation steps, number of grid cells
    // (84 bytes per cell -> 15,000 * 10,000 cells use ~12GB of VRAM)
    uint32_t N_X =      15000;
    uint32_t N_Y =      10000;
    uint32_t N_STEPS =  1;
    uint32_t N_CELLS =  N_X * N_Y;

    // relaxation factor, rest density, max velocity, number of sine periods,
    // wavenumber (frequency)
    float omega = 1.2f;
    float rho_0 = 1.0f;
    float u_max = 0.1f;
    float n = 1.0f;
    float k = (2.0f * PI * n) / static_cast<float>(N_Y);

    // ----- INITIALIZATION OF DISTRIBUTION FUNCTION DATA STRUCTURES -----

    // host-side arrays of 9 pointers to device-side distrib function arrays
    // (used as a host-side handle for the SoA data)
    float* df[9];
    float* df_next[9];

    // for each direction i, allocate 1D array of size N_CELLS on the device
    for (uint32_t i = 0; i < 9; i++)
    {
        cudaMalloc(&df[i], N_CELLS * sizeof(float));
        cudaMalloc(&df_next[i], N_CELLS * sizeof(float));
    }

    // device-side arrays of 9 pointers to device-side distrib function arrays
    // (used as a device-side handle for the SoA data)
    float** dvc_df;
    float** dvc_df_next;
    cudaMalloc(&dvc_df, 9 * sizeof(float*));
    cudaMalloc(&dvc_df_next, 9 * sizeof(float*));

    // copy the contents of the host-side handles to the device-side handle
    cudaMemcpy(dvc_df, df, 9 * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_df_next, df_next, 9 * sizeof(float*), cudaMemcpyHostToDevice);

    // ----- INITIALIZATION OF DENSITY AND VELOCITY DATA STRUCTURES -----

    // pointers to the device-side density and velocity arrays
    float* dvc_rho;
    float* dvc_u_x;
    float* dvc_u_y;

    // for each array, allocate memory of size N_CELLS on the device
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(float));

    // ----- LBM SIMULATION LOOP -----

    Launch_ApplyShearWaveCondition_K(dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0,
        u_max, k, N_X, N_Y, N_CELLS);

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // update densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells using
        // a fully fused kernel performing all core operations in one
        Launch_FullyFusedOperationsComputation(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
            N_CELLS);

        std::swap(dvc_df, dvc_df_next);

        if (step == 1 || step % 100 == 0)
        {
            SPDLOG_INFO("--- step {} done ---", step);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    DisplayPerformanceStats(start_time, end_time, N_X, N_Y, N_STEPS);

    // ----- CLEANUP -----

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
