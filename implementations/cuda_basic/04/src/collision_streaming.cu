#include "../../tools/config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>
#include <stdio.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_csk_c_x[9];
__constant__ int dvc_csk_c_y[9];
__constant__ float dvc_csk_w[9];
bool constantsInitialized_CSK = false;

void InitializeConstants_CSK()
{
    // one-time initialization guard
    if (constantsInitialized_CSK) { return; }

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
    cudaMemcpyToSymbol(dvc_csk_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_csk_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_csk_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized_CSK = true;
}

template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeCollisionAndStreaming_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_next,
    const float* __restrict__ dvc_rho,
    const float* __restrict__ dvc_u_x,
    const float* __restrict__ dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
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

    // ----- COLLISION AND STREAMING COMPUTATION -----

    // load temp variables into read-only cache and multiple loads
    float rho = __ldg(&dvc_rho[idx]);
    float u_x = __ldg(&dvc_u_x[idx]);
    float u_y = __ldg(&dvc_u_y[idx]);
    float u_sq = u_x * u_x + u_y * u_y;

    // determine coordinates of the source cell handled by this thread
    // TODO: bug in coordinate computation?
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X;

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // temp variables for better readability
        float c_x = static_cast<float>(dvc_csk_c_x[i]);
        float c_y = static_cast<float>(dvc_csk_c_y[i]);
        float w = dvc_csk_w[i];

        // dot product of c_i * u (velocity directions times local velocity)
        float cu = c_x * u_x + c_y * u_y;
        float cu2 = cu * cu;

        // compute equilibrium distribution f_eq_i for current direction i
        float f_eq_i = w * rho * (1.0f + 3.0f * cu + 4.5f * cu2 - 1.5f * u_sq);

        // relax distribution function towards equilibrium
        float f_i = df_tile[i][threadIdx.x];
        float f_i_new = f_i + omega * (f_eq_i - f_i);

        // determine coordinates and index within the SoA of the target cell
        // (with respect to periodic boundary conditions)
        uint32_t dst_x = (src_x + dvc_csk_c_x[i] + N_X) % N_X;
        uint32_t dst_y = (src_y + dvc_csk_c_y[i] + N_Y) % N_Y;
        uint32_t dst_idx = dst_y * N_X + dst_x;

        // stream distribution function value df_i to neighbor in direction i
        dvc_df_next[i][dst_idx] = f_i_new;
    }
}

void Launch_CollisionAndStreamingComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    const float* dvc_rho,
    const float* dvc_u_x,
    const float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    InitializeConstants_CSK();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeCollisionAndStreaming_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y, N_CELLS);

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
