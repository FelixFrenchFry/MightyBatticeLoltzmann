#include "../../tools/config.cuh"
#include "../../tools/utilities.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_c_x[9];
__constant__ int dvc_c_y[9];
__constant__ float dvc_w[9];
bool constantsInitialized = false;
bool kernelAttributesDisplayed = false;

void InitializeConstants()
{
    if (constantsInitialized) { return; }

    //  0: ( 0,  0) = rest
    //  1: ( 1,  0) = east
    //  2: ( 0,  1) = north
    //  3: (-1,  0) = west
    //  4: ( 0, -1) = south
    //  5: ( 1,  1) = north-east
    //  6: (-1,  1) = north-west
    //  7: (-1, -1) = south-west
    //  8: ( 1, -1) = south-east

    // initialize velocity direction and weight vectors on the host
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    float w[9] = { 4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
                   1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeFullyFusedOperations_K(
    const float* __restrict__ dvc_df_flat,
    float* __restrict__ dvc_df_next_flat,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // used for summing stuff up and computing collision
    float rho = 0.0f;
    float u_x = 0.0f;
    float u_y = 0.0f;

    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        float df_i = dvc_df_flat[i * N_CELLS + idx];
        rho += df_i;
        u_x += df_i * dvc_c_x[i];
        u_y += df_i * dvc_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= 0.0f) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // TODO: write-back of final density and velocity values necessary?

    // pre-compute squared velocity and cell coordinates for this thread
    float u_sq = u_x * u_x + u_y * u_y;
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        float cu = static_cast<float>(dvc_c_x[i]) * u_x
                 + static_cast<float>(dvc_c_y[i]) * u_y;
        float f_eq_i = dvc_w[i] * rho
                     * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // relax df towards equilibrium
        // TODO: bug in this optimized computation?
        float f_new_i = dvc_df_flat[i * N_CELLS + idx] * (1 - omega) + omega * f_eq_i;

        // determine index of the destination cell within the SoA
        // (with respect to periodic boundary conditions)
        uint32_t dst_idx = ((src_y + dvc_c_y[i] + N_Y) % N_Y) * N_X
                         + ((src_x + dvc_c_x[i] + N_X) % N_X);

        // stream df value df_i to the neighbor in dir i (on the double buffer)
        dvc_df_next_flat[i * N_CELLS + dst_idx] = f_new_i;
    }
}

void Launch_FullyFusedOperationsComputation(
    const float* dvc_df_flat,
    float* dvc_df_next_flat,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df_flat, dvc_df_next_flat, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
        N_CELLS);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed)
    {
        DisplayKernelAttributes(ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE>,
            fmt::format("ComputeFullyFusedOperations_K"),
            N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y);

        kernelAttributesDisplayed = true;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));
    }
}
