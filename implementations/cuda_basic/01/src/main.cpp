// CUDA implementation of Lattice-Boltzmann with notable properties:
// - sequential, non-coalesced memory reads per thread for high cache hit rates
// - separate kernels for density, velocity, collision, streaming operations

#include "../../tools_fp32/data_export.h"
#include "simulation.cuh"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>



int main(int argc, char* argv[])
{
    // configure spdlog to display error messages like this:
    // [hour:min:sec.ms] [file.cpp:line] [type] [message]
    spdlog::set_pattern("[%T.%e] [%s:%#] [%^%l%$] %v");

    // grid size (number of lattice cells per dimension)
    int grid_width =   15000;
    int grid_height =  10000;
    int step_count =   1000;

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
    for (int step = 1; step <= step_count; step++)
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

        if (step % 50 == 0 && false)
        {
            // TODO: fix this
            //ExportSimulationData();

            //SPDLOG_INFO("Exported density data.");
        }

        if (step == 1 || step % 1000 == 0)
        {
            SPDLOG_INFO("--- step {} done ---", step);
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
