#include "../../tools/config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// __restriced__ tells compiler there is no overlap among the data pointed to
// (reduces memory access and instructions, but increases register pressure!)
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeDensityField_K(
    const float* const* __restrict__ dvc_df,
    float* __restrict__ dvc_rho,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // declare and populate df_i in shared memory tile like df_tile[i][thread]
    __shared__ float df_tile[N_DIR][N_BLOCKSIZE];
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        df_tile[i][threadIdx.x] = dvc_df[i][idx];
    }
    // wait for data to be loaded
    __syncthreads();

    float sum_rho = 0.0f;

    // sum over distribution function values in each direction i
    // (SoA layout for coalesced memory access across threads)
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // load data from shared memory tile with local index
        sum_rho += df_tile[i][threadIdx.x];
    }

    dvc_rho[idx] = sum_rho;
}

void Launch_DensityFieldComputation(
    const float* const* dvc_df,
    float* dvc_rho,
    const uint32_t N_CELLS)
{
    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeDensityField_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_rho, N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("Kernel '{}' failed at line {}: {}",
                     __func__, __LINE__, cudaGetErrorString(err));
    }
}
