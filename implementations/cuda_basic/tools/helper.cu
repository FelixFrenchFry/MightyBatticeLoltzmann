#include "config.cuh"
#include "helper.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



__global__ void ComputeVelocityMagnitude_K(
    const float* __restrict__ u_x,
    const float* __restrict__ u_y,
    float* __restrict__ u_mag,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) return;

    float ux = u_x[idx];
    float uy = u_y[idx];
    u_mag[idx] = sqrtf(ux * ux + uy * uy);
}

float* Launch_ComputeVelocityMagnitude_K(
    const float* dvc_u_x,
    const float* dvc_u_y,
    const uint32_t N_X,
    const uint32_t N_Y)
{
    const uint32_t N_CELLS = N_X * N_Y;
    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    // allocate device memory for the velocity magnitudes
    float* dvc_u_mag = nullptr;
    cudaMalloc(&dvc_u_mag, N_CELLS * sizeof(float));

    ComputeVelocityMagnitude_K<<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_u_x, dvc_u_y, dvc_u_mag, N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("Kernel '{}' failed at line {}: {}",
                     __func__, __LINE__, cudaGetErrorString(err));

        cudaFree(dvc_u_mag);
        return nullptr;
    }

    return dvc_u_mag;
}
