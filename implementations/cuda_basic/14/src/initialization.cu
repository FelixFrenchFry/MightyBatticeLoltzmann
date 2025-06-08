#include "../../tools/config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_ik_c_x[9];
__constant__ int dvc_ik_c_y[9];
__constant__ FP dvc_ik_w[9];
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
    FP w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_ik_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_ik_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_ik_w, w, 9 * sizeof(FP));

    cudaDeviceSynchronize();
    constantsInitialized_IK = true;
}

template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ApplyShearWaveCondition_K(
    FP* const* __restrict__ dvc_df,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP rho_0,
    const FP u_max,
    const FP k,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // y-coordinate of the cell handled by this thread
    uint32_t y = idx / N_X;

    // compute sinusoidal x-velocity from the shear wave configuration
    FP u_x_val = u_max * FP_SIN(k * static_cast<FP>(y));
    FP u_sq = u_x_val * u_x_val;

    // set initial values of the fields
    dvc_rho[idx] = rho_0;
    dvc_u_x[idx] = u_x_val;
    dvc_u_y[idx] = FP_CONST(0.0);

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_ik_c_x[i] * u_x_val + dvc_ik_c_y[i] * FP_CONST(0.0);
        FP f_eq_i = dvc_ik_w[i] * rho_0
                  * (FP_CONST(1.0) + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu - FP_CONST(1.5) * u_sq);

        // set initial df values
        dvc_df[i][idx] = f_eq_i;
    }
}

void Launch_ApplyShearWaveCondition_K(
    FP* const* dvc_df,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP rho_0,
    const FP u_max,
    const FP k,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    InitializeConstants_IK();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ApplyShearWaveCondition_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0, u_max, k, N_X, N_Y, N_CELLS);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));
    }
}

template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ApplyLidDrivenCavityCondition_K(
    FP* const* __restrict__ dvc_df,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP rho_0,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // set initial values of the fields
    dvc_rho[idx] = rho_0;
    dvc_u_x[idx] = FP_CONST(0.0);
    dvc_u_y[idx] = FP_CONST(0.0);

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // set initial df values
        dvc_df[i][idx] = dvc_ik_w[i] * rho_0;
    }
}

void Launch_ApplyLidDrivenCavityCondition_K(
    FP* const* dvc_df,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP rho_0,
    const uint32_t N_CELLS)
{
    InitializeConstants_IK();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ApplyLidDrivenCavityCondition_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_rho, dvc_u_x, dvc_u_y, rho_0, N_CELLS);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));
    }
}
