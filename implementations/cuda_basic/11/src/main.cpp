// CUDA implementation of Lattice-Boltzmann using optimization strategies:
// - sequential, non-coalesced memory reads per thread for high cache hit rates
// - shared memory tiles for df values
// - fully fused density/velocity/collision/streaming kernel (push)
// - no global write-back of density and velocity values

#include "../../tools/data_export.h"
#include "fullyfused.cuh"
#include "initialization.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    constexpr float PI = 3.14159265f;

    // TODO: check important hardware properties
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    SPDLOG_INFO("GPU: {} (CC: {})", props.name, props.major * 10 + props.minor);

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

    // pointers to the device-side df, density, and velocity arrays
    float* dvc_df;
    float* dvc_df_next;
    float* dvc_rho;
    float* dvc_u_x;
    float* dvc_u_y;

    // allocate memory on the device
    cudaMalloc(&dvc_df, 9 * N_CELLS * sizeof(float));
    cudaMalloc(&dvc_df_next, 9 * N_CELLS * sizeof(float));
    cudaMalloc(&dvc_rho, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_x, N_CELLS * sizeof(float));
    cudaMalloc(&dvc_u_y, N_CELLS * sizeof(float));

    Launch_ApplyShearWaveCondition_K(dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0,
        u_max, k, N_X, N_Y, N_CELLS);

    for (uint32_t step = 1; step <= N_STEPS; step++)
    {
        // update densities and velocities, update df_i values based on
        // densities and velocities and move them to neighboring cells
        Launch_FullyFusedOperationsComputation(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
            N_CELLS);

        std::swap(dvc_df, dvc_df_next);

        if (step == 1 || step % 100 == 0)
        {
            SPDLOG_INFO("--- step {} done ---", step);
        }
    }

    cudaFree(dvc_df);
    cudaFree(dvc_df_next);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    return 0;
}
