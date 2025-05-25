#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// __restriced__ tells compiler there is no overlap among the data pointed to
// (reduces memory access and instructions, but increases register pressure!)
template <int N_DIR> // specify loop count at compile time for optimizations
__global__ void ComputeDensityField_K(
    const float* const* __restrict__ dvc_df,
    float* __restrict__ dvc_rho,
    const size_t N_CELLS)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    float sum_rho = 0.0f;

    // sum over distribution function values in each direction i
    // (SoA layout for coalesced memory access across threads)
    #pragma unroll
    for (int i = 0; i < N_DIR; i++)
    {
        sum_rho += dvc_df[i][idx];
    }

    dvc_rho[idx] = sum_rho;
}

void Launch_DensityFieldComputation(
    const float* const* dvc_df,
    float* dvc_rho,
    const size_t N_CELLS)
{
    const int blockSize = 256;
    const int gridSize = (N_CELLS + blockSize - 1) / blockSize;

    ComputeDensityField_K<9><<<gridSize, blockSize>>>(
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
