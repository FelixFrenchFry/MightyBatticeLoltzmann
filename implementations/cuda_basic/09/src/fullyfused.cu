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
template <uint32_t N_DIR, uint32_t N_TILE_X, uint32_t N_TILE_Y>
__global__ void __launch_bounds__(N_TILE_X * N_TILE_Y, 6)
ComputeFullyFusedOperations_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_next,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS)
{
    const uint32_t x_local = threadIdx.x;
    const uint32_t y_local = threadIdx.y;

    // determine coordinates of this thread's own cell
    // (destination of the df_i values pulled from the neighbors)
    const uint32_t dst_x = blockIdx.x * N_TILE_X + x_local;
    const uint32_t dst_y = blockIdx.y * N_TILE_Y + y_local;
    if (dst_x >= N_X || dst_y >= N_Y) return;

    const uint32_t idx_global = dst_y * N_X + dst_x;

    // shared memory tile for df values with 1-layer halo cells in each direction
    // (filled via coalesced accesses, read from via non-coalesced accesses)
    // (with 16x16 threads per block -> ~11,4KB of shared memory per block)
    __shared__ float df_tile[N_DIR][N_TILE_Y + 2][N_TILE_X + 2];

    // threads cooperatively populate tile with df values from global memory
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // TODO: do this properly coalesced and efficent (and ideally, correct aswell)

        for (uint32_t dy = -1; dy <= 1; dy++)
        {
            const uint32_t load_y = y_local + 1 + dy;
            uint32_t src_y = (dst_y + dy + N_Y) % N_Y;

            for (uint32_t dx = -1; dx <= 1; dx++)
            {
                const uint32_t load_x = x_local + 1 + dx;
                uint32_t src_x = (dst_x + dx + N_X) % N_X;
                uint32_t src_idx = src_y * N_X + src_x;

                if (true)
                {
                    df_tile[i][load_y][load_x] = dvc_df[i][src_idx];
                }
            }
        }
    }
    /*
    #pragma unroll
    for (uint32_t i = 0; i < 5; i++)
    {
        // direction vector
        int cx = dvc_c_x[i];
        int cy = dvc_c_y[i];

        // offset into shared memory tile (1 for halo)
        uint32_t load_x = x_local + 1 + cx;
        uint32_t load_y = y_local + 1 + cy;

        // global coordinates to load (with periodic BC)
        uint32_t src_x = (dst_x + cx + N_X) % N_X;
        uint32_t src_y = (dst_y + cy + N_Y) % N_Y;
        uint32_t src_idx = src_y * N_X + src_x;

        // cooperative load into shared memory
        df_tile[i][load_y][load_x] = dvc_df[i][src_idx];
    }
    */
    // wait for all threads blocks to finish loading the tile
    __syncthreads();

    // ----- STREAMING COMPUTATION (PULL) -----
    // ----- FUZED WITH DENSITY AND VELOCITY COMPUTATION -----

    // temp storage of df_i values pulled from neighbors
    // TODO: use shared memory for this?
    float df[9];

    // used initially as sum and later as final values in computations
    float rho = 0.0f;
    float u_x = 0.0f;
    float u_y = 0.0f;

    // pull df_i values from each neighbor in direction i
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute coordinates of the source neighbor in direction i,
        // from which to pull the df_i value (with peridic boundary conditions)
        uint32_t src_x = 1 + threadIdx.x - dvc_c_x[i];
        uint32_t src_y = 1 + threadIdx.y - dvc_c_y[i];

        // pull df_i from the neighbor in direction i
        // (is memory access coalesced with neighboring threads?)
        float df_i = df_tile[i][src_y][src_x];
        df[i] = df_i;

        // sum over df_i values to compute density and velocities
        rho += df_i;
        u_x += df_i * dvc_c_x[i];
        u_y += df_i * dvc_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= 0.0f)
    {
        dvc_rho[idx_global] = 0.0f;
        dvc_u_x[idx_global] = 0.0f;
        dvc_u_y[idx_global] = 0.0f;
        return;
    }

    // divide sums by density to obtain final velocities
    u_x /= rho;
    u_y /= rho;

    // write back updated values to global memory
    dvc_rho[idx_global] = rho;
    dvc_u_x[idx_global] = u_x;
    dvc_u_y[idx_global] = u_y;

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
        dvc_df_next[i][idx_global] = f_new_i;
    }
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

    // define dimensions of 2D block of threads and 2D grid of blocks
    dim3 blockDimensions(N_TILE_X, N_TILE_Y);
    dim3 gridDimensions((N_X + N_TILE_X - 1) / N_TILE_X,
                        (N_Y + N_TILE_Y - 1) / N_TILE_Y);

    ComputeFullyFusedOperations_K<N_DIR, N_TILE_X, N_TILE_Y>
        <<<gridDimensions, blockDimensions>>>(
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
