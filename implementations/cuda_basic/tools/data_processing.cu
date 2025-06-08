#include "config.cuh"
#include "data_processing.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



__global__ void ComputeVelocityMagnitude_K(
    const FP* __restrict__ u_x,
    const FP* __restrict__ u_y,
    FP* __restrict__ u_mag,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) return;

    FP ux = u_x[idx];
    FP uy = u_y[idx];
    u_mag[idx] = FP_SQRT(ux * ux + uy * uy);
}

FP* Launch_ComputeVelocityMagnitude_K(
    const FP* dvc_u_x,
    const FP* dvc_u_y,
    const uint32_t N_X,
    const uint32_t N_Y)
{
    const uint32_t N_CELLS = N_X * N_Y;
    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    // allocate device memory for the velocity magnitudes
    FP* dvc_u_mag = nullptr;
    cudaMalloc(&dvc_u_mag, N_CELLS * sizeof(FP));

    ComputeVelocityMagnitude_K<<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_u_x, dvc_u_y, dvc_u_mag, N_CELLS);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));

        cudaFree(dvc_u_mag);
        return nullptr;
    }

    return dvc_u_mag;
}
