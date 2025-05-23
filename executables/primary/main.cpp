// primary, optimized CUDA implementation of the Lattice-Boltzmann method

#include "collision.cuh"
#include "density.cuh"
#include "initialization.cuh"
#include "output/export.h"
#include "streaming.cuh"
#include "velocity.cuh"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

int main(int argc, char* argv[])
{
    // ----- INITIALIZATION OF MISC STUFF -----

    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    constexpr float PI = 3.14159265f;

    // ----- INITIALIZATION OF PARAMETERS -----

    // grid width, height, number of simulation steps, number of grid cells
    // (15,000 * 10,000 cells use ~12GB of VRAM)
    size_t N_X =        150;
    size_t N_Y =        100;
    size_t N_STEPS =    3;
    size_t N_CELLS = N_X * N_Y;

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
    for (size_t i = 0; i < 9; i++)
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

    for (size_t step = 1; step <= N_STEPS; step++)
    {
        // update densities
        Launch_DensityFieldComputation(
            dvc_df, dvc_rho, N_CELLS);

        // update velocities
        Launch_VelocityFieldComputation(
            dvc_df, dvc_rho, dvc_u_x, dvc_u_y, N_CELLS);

        // update df_i values based on densities and velocities
        Launch_CollisionComputation(
            dvc_df, dvc_rho, dvc_u_x, dvc_u_y, omega, N_CELLS);

        // move updated df_i values to neighboring cells
        Launch_StreamingComputation(
            dvc_df, dvc_df_next, N_X, N_Y, N_CELLS);

        // TODO: compare performance of fused collision + streaming kernel

        std::swap(dvc_df, dvc_df_next);

        SPDLOG_INFO("--- step {} done ---", step);
    }

    // ----- CLEANUP -----

    for (size_t i = 0; i < 9; i++)
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
