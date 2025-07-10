#include "../../tools/config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int con_ik_c_x[9];
__constant__ int con_ik_c_y[9];
__constant__ float con_ik_w[9];
bool constantsInitialized_IK = false;

void InitializeConstants_IK()
{
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
    float w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                   1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(con_ik_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_ik_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_ik_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized_IK = true;
}

template <int N_BLOCKSIZE>
__global__ void ApplyInitialCondition_ShearWaveDecay_K(
    float* const* __restrict__ dvc_df,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float rho_0,
    const float u_max,
    const float w_num,
    const int N_X, const int N_Y,
    const int Y_START,
    const int N_CELLS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // y-coordinate of the cell handled by this thread
    int y_global = (idx / N_X) + Y_START;

    // compute sinusoidal x-velocity from the shear wave configuration
    float u_x_val = u_max * sinf(w_num * static_cast<float>(y_global));
    float u_sq = u_x_val * u_x_val;

    // set initial values of the fields
    dvc_rho[idx] = rho_0;
    dvc_u_x[idx] = u_x_val;
    dvc_u_y[idx] = 0.0f;

    for (int i = 0; i < 9; i++)
    {
        // dot product of c_i * u
        float cu = con_ik_c_x[i] * u_x_val + con_ik_c_y[i] * 0.0f;

        // equilibrium df value for dir i
        float f_eq_i = con_ik_w[i] * rho_0 * (1.0f
                     + 3.0f * cu
                     + 4.5f * cu * cu
                     - 1.5f * u_sq);

        // set initial df values
        dvc_df[i][idx] = f_eq_i;
    }
}

void Launch_ApplyInitialCondition_ShearWaveDecay_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const float u_max,
    const float w_num,
    const int N_X, const int N_Y,
    const int Y_START,
    const int N_CELLS)
{
    InitializeConstants_IK();

    const int N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ApplyInitialCondition_ShearWaveDecay_K<N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0, u_max, w_num, N_X, N_Y, Y_START, N_CELLS);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // use detailed logging format for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic format
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }
}

template <int N_BLOCKSIZE>
__global__ void ApplyInitialCondition_LidDrivenCavity_K(
    float* const* __restrict__ dvc_df,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float rho_0,
    const int N_CELLS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // set initial values of the fields
    dvc_rho[idx] = rho_0;
    dvc_u_x[idx] = 0.0f;
    dvc_u_y[idx] = 0.0f;

    for (int i = 0; i < 9; i++)
    {
        // set initial df values
        dvc_df[i][idx] = con_ik_w[i] * rho_0;
    }
}

void Launch_ApplyInitialCondition_LidDrivenCavity_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const int N_CELLS)
{
    InitializeConstants_IK();

    const int N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ApplyInitialCondition_LidDrivenCavity_K<N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0, N_CELLS);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // use detailed logging format for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic format
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }
}
