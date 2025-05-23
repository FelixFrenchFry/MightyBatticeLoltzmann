// primary, optimized CUDA implementation of the Lattice-Boltzmann method

#include "collision.cuh"
#include "density.cuh"
#include "output/export.h"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "simulation.cuh"
#include "velocity.cuh"



int main(int argc, char* argv[])
{
    // ----- INITIALIZATION OF MISC STUFF -----

    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    constexpr float PI = 3.14159265f;

    // ----- INITIALIZATION OF PARAMETERS -----

    // grid width, height, number of simulation steps, number of grid cells
    size_t N_X =        60;
    size_t N_Y =        40;
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

    // host-side array of 9 pointers to device-side distrib function arrays
    // (used as a host-side handle for the SoA data)
    float* df[9];

    // for each direction dir, allocate 1D array of size N_CELLS on the device
    for (size_t dir = 0; dir < 9; dir++)
    {
        cudaMalloc(&df[dir], N_CELLS * sizeof(float));
    }

    // device-side array of 9 pointers to device-side distrib function arrays
    // (used as a device-side handle for the SoA data)
    float** dvc_df;
    cudaMalloc(&dvc_df, 9 * sizeof(float*));

    // copy the contents of the host-side handle to the device-side handle
    cudaMemcpy(dvc_df, df, 9 * sizeof(float*), cudaMemcpyHostToDevice);

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

    for (size_t step = 1; step <= N_STEPS; step++)
    {
        Launch_DensityFieldComputation_temp(
            dvc_df, dvc_rho, N_CELLS);

        Launch_VelocityFieldComputation_temp(
            dvc_df, dvc_rho, dvc_u_x, dvc_u_y, N_CELLS);

        Launch_CollisionComputation_temp(
            dvc_df, dvc_rho, dvc_u_x, dvc_u_y, omega, N_CELLS);

        SPDLOG_INFO("--- step {} done ---", step);
    }

    // ----- CLEANUP -----

    for (size_t i = 0; i < 9; i++)
    {
        cudaFree(df[i]);
    }
    cudaFree(dvc_df);
    cudaFree(dvc_rho);
    cudaFree(dvc_u_x);
    cudaFree(dvc_u_y);

    return 0;

    // -------------------------------------------------------------------------





    // -------------------------------------------------------------------------

    // grid size (number of lattice cells per dimension)
    int grid_width =   300;
    int grid_height =  200;

    // misc
    float relaxOmega = 1.2f;
    float restDensity = 1.0f;
    int num_cells = grid_width * grid_height;
    int num_dirs = 9;

    // initialize the velocity vectors sitting on constant memory on the device
    Initialize();

    // create pointers for current and next distribution function f_i(x,y)
    // stored as a 1D linear buffer on the device and allocate memory on it
    float* dvc_distributionFunc =       nullptr;
    float* dvc_distributionFunc_next =  nullptr;

    cudaMalloc(&dvc_distributionFunc, num_cells * num_dirs * sizeof(float));
    cudaMalloc(&dvc_distributionFunc_next, num_cells * num_dirs * sizeof(float));

    // create pointers for the density values and velocity vectors stored
    // on the device and allocate memory on it
    float* dvc_densityField =           nullptr;
    float* dvc_velocityField_x =        nullptr;
    float* dvc_velocityField_y =        nullptr;

    cudaMalloc(&dvc_densityField, num_cells * sizeof(float));
    cudaMalloc(&dvc_velocityField_x, num_cells * sizeof(float));
    cudaMalloc(&dvc_velocityField_y, num_cells * sizeof(float));

    // launch kernel for initializing the distribution function
    Launch_InitializeDistributionFunction_K(
        dvc_distributionFunc, 1.0f, num_cells * num_dirs);

    // launch kernel for initializing the density field
    Launch_InitializeDensityField_K(
        dvc_densityField, restDensity, num_cells);

    // alternatively, use optimized CUDA function to initialize values to zero
    // cudaMemset(dvc_distributionFunc, 0, num_cells * num_dirs * sizeof(float));

    // run the (incomplete) simulation step for a specified number of iterations
    for (int step = 0; step < 200; step++)
    {
        // launch kernel for computing the density field
        Launch_ComputeDensityField_K(
            dvc_distributionFunc, dvc_densityField, num_cells);

        // launch kernel for computing the velocity field
        Launch_ComputeVelocityField_K(
            dvc_distributionFunc, dvc_densityField, dvc_velocityField_x,
            dvc_velocityField_y, num_cells);

        // launch kernel for the collision step
        Launch_CollisionStep_K(
            dvc_distributionFunc, dvc_densityField, dvc_velocityField_x,
            dvc_velocityField_y, relaxOmega, num_cells);

        // launch kernel for the streaming step
        Launch_StreamingStep_K(
            dvc_distributionFunc, dvc_distributionFunc_next, grid_width,
            grid_height, num_cells);

        std::swap(dvc_distributionFunc, dvc_distributionFunc_next);

        if (step % 50 == 0)
        {
            ExportScalarField(dvc_densityField, num_cells,
                "density" + std::to_string(step) + ".bin");

            SPDLOG_INFO("Exported density data.");
        }
    }

    // free device memory
    cudaFree(dvc_distributionFunc);
    cudaFree(dvc_distributionFunc_next);
    cudaFree(dvc_densityField);
    cudaFree(dvc_velocityField_x);
    cudaFree(dvc_velocityField_y);

    return 0;
}
