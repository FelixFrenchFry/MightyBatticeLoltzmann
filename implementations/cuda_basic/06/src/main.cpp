// CUDA implementation of Lattice-Boltzmann with notable properties:
// - coalesced memory accesses of df values
// - vectorized memory transfers (TODO: unfinished, bugs)
// - shared memory tiling for df values
// - fully fused kernel for density/velocity/collision/streaming operations

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

    // required for float4 vectorized memory accesses
    static_assert(N_VECSIZE == 4);
    assert(N_CELLS % N_VECSIZE == 0);

    // relaxation factor, rest density, max velocity, number of sine periods,
    // wavenumber (frequency)
    float omega = 1.2f;
    float rho_0 = 1.0f;
    float u_max = 0.1f;
    float n = 1.0f;
    float k = (2.0f * PI * n) / static_cast<float>(N_Y);

    // ----- INITIALIZATION OF DISTRIBUTION FUNCTION DATA STRUCTURES -----

    DF_Vec* dvc_df_1_to_8;
    DF_Vec* dvc_df_next_1_to_8;

    cudaMalloc(&dvc_df_1_to_8, N_CELLS * sizeof(DF_Vec));
    cudaMalloc(&dvc_df_next_1_to_8, N_CELLS * sizeof(DF_Vec));

    assert(reinterpret_cast<uintptr_t>(dvc_df_1_to_8) % alignof(float4) == 0);
    assert(reinterpret_cast<uintptr_t>(dvc_df_next_1_to_8) % alignof(float4) == 0);

    // pointer to the device-side center direction
    // (separate from the struct to avoid dummy values required for alignment)
    // TODO: is double buffering required for the streaming step?
    float* dvc_df_0;

    cudaMalloc(&dvc_df_0, N_CELLS * sizeof(float));

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

    Launch_ApplyShearWaveCondition_K(dvc_df_1_to_8, dvc_df_0, dvc_rho, dvc_u_x,
        dvc_u_y, rho_0, u_max, k, N_X, N_Y, N_CELLS);

    auto start_time = std::chrono::steady_clock::now();

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // update densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells using
        // a fully fused kernel performing all core operations in one
        Launch_FullyFusedOperationsComputation(
            dvc_df_1_to_8, dvc_df_next_1_to_8, dvc_df_0, dvc_rho, dvc_u_x,
            dvc_u_y, omega, N_X, N_Y, N_CELLS);

        std::swap(dvc_df_1_to_8, dvc_df_next_1_to_8);

        if (step == 1 || step % 100 == 0)
        {
            SPDLOG_INFO("--- step {} done ---", step);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    DisplayPerformanceStats(start_time, end_time, N_X, N_Y, N_STEPS);

    // ----- CLEANUP -----

    cudaFree(dvc_df_1_to_8);
    cudaFree(dvc_df_next_1_to_8);
    cudaFree(dvc_df_0);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    return 0;
}
