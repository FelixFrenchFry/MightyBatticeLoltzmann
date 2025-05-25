#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_ik_c_x[9];
__constant__ int dvc_ik_c_y[9];
__constant__ float dvc_ik_w[9];
bool constantsInitialized_IK = false;

void InitializeConstants_IK()
{
    // one-time initialization guard
    if (constantsInitialized_IK) { return; }

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
    cudaMemcpyToSymbol(dvc_ik_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_ik_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_ik_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized_IK = true;
}

template <int N_DIR> // specify loop count at compile time for optimizations
__global__ void ApplyShearWaveCondition_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const float u_max,
    const float k,
    const size_t N_X, const size_t N_Y,
    const size_t N_CELLS)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // determine coordinates of the cell handled by this thread
    int x = idx % N_X;
    int y = idx / N_X;

    // compute sinusoidal x-velocity from the shear wave configuration
    float u_x_val = u_max * sinf(k * static_cast<float>(y));
    float u_sq = u_x_val * u_x_val;

    // set initial values of different fields
    dvc_rho[idx] = rho_0;
    dvc_u_x[idx] = u_x_val;
    dvc_u_y[idx] = 0.0f;

    // initialize distribution function values according to an equilibrium
    #pragma unroll
    for (int i = 0; i < N_DIR; i++)
    {
        // dot product of c_i * u (velocity directions times local velocity)
        float cu = dvc_ik_c_x[i] * u_x_val + dvc_ik_c_y[i] * 0.0f;

        // compute equilibrium distribution f_eq_i for current direction i
        float f_eq_i = dvc_ik_w[i] * rho_0
            * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        dvc_df[i][idx] = f_eq_i;
    }
}

void Launch_ApplyShearWaveCondition_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const float u_max,
    const float k,
    const size_t N_X, const size_t N_Y,
    const size_t N_CELLS)
{
    InitializeConstants_IK();

    const int blockSize = 256;
    const int gridSize = (N_CELLS + blockSize - 1) / blockSize;

    ApplyShearWaveCondition_K<9><<<gridSize, blockSize>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0, u_max, k, N_X, N_Y, N_CELLS);

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
