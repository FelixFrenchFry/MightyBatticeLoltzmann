#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_ck_c_x[9];
__constant__ int dvc_ck_c_y[9];
__constant__ float dvc_ck_w[9];
bool constantsInitialized_CK = false;

void InitializeConstants_CK()
{
    // one-time initialization guard
    if (constantsInitialized_CK) { return; }

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
    cudaMemcpyToSymbol(dvc_ck_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_ck_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_ck_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized_CK = true;
}

__global__ void ComputeCollision_K_temp(
    float* const* __restrict__ dvc_df,
    const float* __restrict__ dvc_rho,
    const float* __restrict__ dvc_u_x,
    const float* __restrict__ dvc_u_y,
    const float omega,
    const size_t N_CELLS)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // temp variables to avoid multiple loads from memory
    float rho = dvc_rho[idx];
    float u_x = dvc_u_x[idx];
    float u_y = dvc_u_y[idx];
    float u_sq = u_x * u_x + u_y * u_y;

    #pragma unroll
    for (int i = 0; i < 9; i++)
    {
        // temp variables for better readability (and less loads from memory)
        float c_x = static_cast<float>(dvc_ck_c_x[i]);
        float c_y = static_cast<float>(dvc_ck_c_y[i]);
        float w = dvc_ck_w[i];

        // dot product of c_i * u (velocity directions times local velocity)
        float cu = c_x * u_x + c_y * u_y;

        // compute equilibrium distribution f_eq_i for current direction i
        float f_eq_i = w * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // relax distribution function towards equilibrium
        float f_i = dvc_df[i][idx];
        dvc_df[i][idx] = f_i + omega * (f_eq_i - f_i);
    }
}

void Launch_CollisionComputation_temp(
    float* const* dvc_df,
    const float* dvc_rho,
    const float* dvc_u_x,
    const float* dvc_u_y,
    const float omega,
    const size_t N_CELLS)
{
    const int blockSize = 256;
    const int gridSize = (N_CELLS + blockSize - 1) / blockSize;

    ComputeCollision_K_temp<<<gridSize, blockSize>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, omega, N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA collision kernel error: {}",
            cudaGetErrorString(err));
    }
}
