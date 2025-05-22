#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// __restriced__ tells compiler there is no overlap among the data pointed to
// (reduces memory access and instructions, but increases register pressure!)
__global__ void ComputeVelocityField_K_temp(
    const float* const* __restrict__ dvc_df,
    const float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const size_t N_CELLS)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // store density in temp variable to avoid multiple loads from memory
    float local_rho = dvc_rho[idx];

    // exit thread to avoid division by zero or erroneous values
    if (local_rho <= 0.0f)
    {
        dvc_u_x[idx] = 0.0f;
        dvc_u_y[idx] = 0.0f;
        return;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    // sum over distribution function values, weighted by each direction
    // (SoA layout for coalesced memory access across threads)
    #pragma unroll
    for (int dir = 0; dir < 9; dir++)
    {
        float f_i = dvc_df[idx][dir];
        sum_x += f_i * dvc_c_x[dir];
        sum_y += f_i * dvc_c_y[dir];
    }

    // divide sums by density to obtain final velocities
    dvc_u_x[idx] = sum_x / local_rho;
    dvc_u_y[idx] = sum_y / local_rho;
}

void Launch_VelocityFieldComputation_temp(
    const float* const* dvc_df,
    const float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const size_t N_CELLS)
{
    const int blockSize = 256;
    const int gridSize = (N_CELLS + blockSize - 1) / blockSize;

    ComputeVelocityField_K_temp<<<gridSize, blockSize>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA velocity kernel error: {}", cudaGetErrorString(err));
    }
}
