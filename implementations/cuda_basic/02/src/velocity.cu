#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



// load velocity direction vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_vk_c_x[9];
__constant__ int dvc_vk_c_y[9];
bool constantsInitialized_VK = false;

void InitializeConstants_VK()
{
    // one-time initialization guard
    if (constantsInitialized_VK) { return; }

    //  0: ( 0,  0) = rest
    //  1: ( 1,  0) = east
    //  2: ( 0,  1) = north
    //  3: (-1,  0) = west
    //  4: ( 0, -1) = south
    //  5: ( 1,  1) = north-east
    //  6: (-1,  1) = north-west
    //  7: (-1, -1) = south-west
    //  8: ( 1, -1) = south-east

    // initialize velocity direction vectors on the host
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_vk_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_vk_c_y, c_y, 9 * sizeof(int));

    cudaDeviceSynchronize();
    constantsInitialized_VK = true;
}

// __restriced__ tells compiler there is no overlap among the data pointed to
// (reduces memory access and instructions, but increases register pressure!)
template <int N_DIR> // specify loop count at compile time for optimizations
__global__ void ComputeVelocityField_K(
    const float* const* __restrict__ dvc_df,
    const float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const size_t N_CELLS)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // store density in temp variable to avoid multiple loads from memory
    float rho = dvc_rho[idx];

    // exit thread to avoid division by zero or erroneous values
    if (rho <= 0.0f)
    {
        dvc_u_x[idx] = 0.0f;
        dvc_u_y[idx] = 0.0f;
        return;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    // sum over distribution function values, weighted by each direction i
    // (SoA layout for coalesced memory access across threads)
    #pragma unroll
    for (int i = 0; i < N_DIR; i++)
    {
        float df_i = dvc_df[i][idx];
        sum_x += df_i * dvc_vk_c_x[i];
        sum_y += df_i * dvc_vk_c_y[i];
    }

    // divide sums by density to obtain final velocities
    dvc_u_x[idx] = sum_x / rho;
    dvc_u_y[idx] = sum_y / rho;
}

void Launch_VelocityFieldComputation(
    const float* const* dvc_df,
    const float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const size_t N_CELLS)
{
    InitializeConstants_VK();

    const int blockSize = 256;
    const int gridSize = (N_CELLS + blockSize - 1) / blockSize;

    ComputeVelocityField_K<9><<<gridSize, blockSize>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA velocity kernel error: {}",
            cudaGetErrorString(err));
    }
}
