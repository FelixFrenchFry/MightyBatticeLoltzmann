#include "../../tools/config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load velocity direction and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
__constant__ int dvc_c_x[9];
__constant__ int dvc_c_y[9];
__constant__ float dvc_w[9];
bool constantsInitialized = false;

void InitializeConstants()
{
    // one-time initialization guard
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

// __restriced__ tells compiler there is no overlap among the data pointed to
// (reduces memory access and instructions, but increases register pressure!)
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeFullyFusedOperations_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_next,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // ----- STREAMING COMPUTATION (PULL) -----

    // determine coordinates of this thread's own cell
    // (destination of the df_i values pulled from the neighbors)
    uint32_t dst_x = idx % N_X;
    uint32_t dst_y = idx / N_X;

    // temp storage of df_i values pulled from neighbors
    // TODO: use shared memory for this?
    float df[9];

    // pull df_i values from each neighbor in direction i
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute coordinates of the source neighbor in direction i,
        // from which to pull the df_i value
        uint32_t src_x = static_cast<int>(dst_x) - dvc_c_x[i];
        uint32_t src_y = static_cast<int>(dst_y) - dvc_c_y[i];

        // apply periodic boundary conditions
        src_x = (src_x + N_X) % N_X;
        src_y = (src_y + N_Y) % N_Y;
        uint32_t src_idx = src_y * N_X + src_x;

        // pull df_i from the neighbor in direction i
        // (is memory access coalesced with neighboring threads?)
        // TODO: store this in shared memory instead of relying on registers?
        df[i] = dvc_df[i][src_idx];
    }

    // ----- DENSITY COMPUTATION -----

    // used initially as sum and later as final density in computations
    float rho = 0.0f;

    // sum over df_i values to compute density
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        rho += df[i];
    }

    // ----- VELOCITY COMPUTATION -----

    // exit thread to avoid division by zero or erroneous values
    if (rho <= 0.0f)
    {
        dvc_u_x[idx] = 0.0f;
        dvc_u_y[idx] = 0.0f;
        return;
    }

    // used initially as sums and later as final velocities in computations
    float u_x = 0.0f;
    float u_y = 0.0f;

    // sum over df_i values, weighted in each direction to compute velocity
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        u_x += df[i] * dvc_c_x[i];
        u_y += df[i] * dvc_c_y[i];
    }

    // divide sums by density to obtain final velocities
    u_x /= rho;
    u_y /= rho;

    // ----- COLLISION COMPUTATION -----

    // pre-compute squared velocity of this thread's cell
    float u_sq = u_x * u_x + u_y * u_y;

    // update df_i values by relaxation towards equilibrium
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // dot product of c_i * u (velocity directions times local velocity)
        float cu = static_cast<float>(dvc_c_x[i]) * u_x
                 + static_cast<float>(dvc_c_y[i]) * u_y;

        // compute equilibrium distribution f_eq_i for direction i
        float f_eq_i = dvc_w[i] * rho
                     * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // relax distribution function towards equilibrium
        float f_new_i = df[i] * (1 - omega) + omega * f_eq_i;

        // update df value of this thread's cell in global memory
        dvc_df_next[i][idx] = f_new_i;
    }

    // ----- MISC -----

    // write back updated values to global memory
    dvc_rho[idx] = rho;
    dvc_u_x[idx] = u_x;
    dvc_u_y[idx] = u_y;
}

void Launch_FullyFusedOperationsComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
        N_CELLS);

    // wait for device actions to finish and report potential errors
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed)
    {
        DisplayKernelAttributes(ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE>,
            fmt::format("ComputeFullyFusedOperations_K<{}, {}>", N_DIR, N_BLOCKSIZE));

        kernelAttributesDisplayed = true;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("Kernel '{}' failed: {}",
                     __func__, cudaGetErrorString(err));
    }
}
