#include "../../tools/config.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load opposite direction vectors for bounce-back, velocity direction vectors,
// and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_opp_dir[9];
__constant__ int dvc_c_x[9];
__constant__ int dvc_c_y[9];
__constant__ FP dvc_fp_c_x[9];
__constant__ FP dvc_fp_c_y[9];
__constant__ FP dvc_w[9];
bool constantsInitialized = false;

void InitializeConstants()
{
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

    // initialize opposite direction, velocity direction, and weight vectors
    int opp_dir[9] = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    FP fp_c_x[9] = { 0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0 };
    FP fp_c_y[9] = { 0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0 };
    FP w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_opp_dir, opp_dir, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_fp_c_x, fp_c_x, 9 * sizeof(FP));
    cudaMemcpyToSymbol(dvc_fp_c_y, fp_c_y, 9 * sizeof(FP));
    cudaMemcpyToSymbol(dvc_w, w, 9 * sizeof(FP));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

template <uint32_t N_DIR>
__device__ __forceinline__ uint32_t ComputeDensityAndVelocity_K(
    const FP tile_df[N_DIR][N_BLOCKSIZE],
    uint32_t tid,
    FP& rho, FP& u_x, FP& u_y)
{
    rho = FP_CONST(0.0);
    u_x = FP_CONST(0.0);
    u_y = FP_CONST(0.0);

    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        FP df_i = tile_df[i][tid];
        rho += df_i;
        u_x += df_i * dvc_fp_c_x[i];
        u_y += df_i * dvc_fp_c_y[i];
    }
}

__device__ __forceinline__ void ComputeNeighborIndex_PeriodicBoundary_K(
    uint32_t src_x, uint32_t src_y,
    uint32_t N_X, uint32_t N_Y,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // determine index of the destination neihbor cell
    // (with respect to periodic boundary conditions)
    dst_idx = ((src_y + dvc_c_y[i] + N_Y) % N_Y) * N_X
            + ((src_x + dvc_c_x[i] + N_X) % N_X);
    dst_i = i;
}

__device__ __forceinline__ void ComputeNeighborIndex_BounceBackBoundary_Conditional_K(
    uint32_t src_x, uint32_t src_y,
    uint32_t N_X, uint32_t N_Y,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // check if directed into a wall
    if ((dvc_c_x[i] == -1 && src_x == 0) ||        // into left wall
        (dvc_c_x[i] ==  1 && src_x == N_X - 1) ||  // into right wall
        (dvc_c_y[i] == -1 && src_y == 0) ||        // into bottom wall
        (dvc_c_y[i] ==  1 && src_y == N_Y - 1))    // into top wall
    {
        // same cell but opposite direction because of bounce-back
        dst_idx = src_y * N_X + src_x;
        dst_i = dvc_opp_dir[i];
    }
    else
    {
        // normal neighbor in direction i
        dst_idx = (src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]);
        dst_i = i;
    }
}

__device__ __forceinline__ void ComputeNeighborIndex_BounceBackBoundary_BranchLess_K(
    uint32_t src_x, uint32_t src_y,
    uint32_t N_X, uint32_t N_Y,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // TODO: this increases register pressure by too much

    // branch-less bit-wise computation of: 1 if bounce-back, else 0
    int bounce =
        ((dvc_c_x[i] == -1) & (src_x == 0)) |
        ((dvc_c_x[i] ==  1) & (src_x == N_X - 1)) |
        ((dvc_c_y[i] == -1) & (src_y == 0)) |
        ((dvc_c_y[i] ==  1) & (src_y == N_Y - 1));

    // branch-less computation of destination index
    // TODO: reduce register usage
    /*
    uint32_t idx_normal = (src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]);
    uint32_t idx_bounce = src_y * N_X + src_x;
    dst_idx = bounce * idx_bounce + (1 - bounce) * idx_normal;
    */
    dst_idx = bounce * (src_y * N_X + src_x)
            + (1 - bounce) * ((src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]));

    // branch-less computation of destination direction
    dst_i = bounce * dvc_opp_dir[i] + (1 - bounce) * i;
}

__device__ __forceinline__ void InjectLidVelocity_Conditional_K(
    uint32_t src_y,
    uint32_t N_Y,
    FP rho,
    FP omega,
    FP u_lid,
    uint32_t i,
    FP& f_new_i)
{
    // check if directed into top wall
    if (dvc_c_y[i] == 1 && src_y == N_Y - 1)
    {
        f_new_i -= FP_CONST(6.0) * omega * rho * dvc_fp_c_x[i] * u_lid;
    }
}

__device__ __forceinline__ void InjectLidVelocity_BranchLess_K(
    uint32_t src_y,
    uint32_t N_Y,
    FP rho,
    FP omega,
    FP u_lid,
    uint32_t i,
    FP& f_new_i)
{
    // branch-less lid velocity injection via boolean mask
    int top_bounce = ((dvc_c_y[i] ==  1) & (src_y == N_Y - 1));
    f_new_i -= top_bounce * FP_CONST(6.0) * dvc_w[i] * rho
             * dvc_fp_c_x[i] * u_lid;
}

template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void ComputeFullyFusedOperations_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;
    if (idx >= N_CELLS) { return; }

    // load df values into tiles of shared shared memory
    // TODO: add 1-layer halo cells?
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // determine coordinates of this thread's own cell
    // (destination of the df values pulled from the neighbors)
    uint32_t dst_x = idx % N_X;
    uint32_t dst_y = idx / N_X;

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

        // pull df value from neighbor in i direction and store in shared memory
        // (is memory access properly coalesced with neighboring threads?)
        tile_df[i][tid] = dvc_df[i][src_idx];
    }

    // inlined sub-kernel for the density and velocity
    FP rho, u_x, u_y;
    ComputeDensityAndVelocity_K<N_DIR>(
        tile_df, tid, rho, u_x, u_y);

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity and cell coordinates for this thread
    FP u_sq = u_x * u_x + u_y * u_y;

    #pragma unroll
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho
                  * (FP_CONST(1.0) + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][tid] - omega
                   * (tile_df[i][tid] - f_eq_i);

        // update df value of this thread's cell in global memory
        dvc_df_next[i][idx] = f_new_i;
    }
}

void Launch_FullyFusedOperationsComputation(
    const FP* const* dvc_df,
    FP* const* dvc_df_next,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    ComputeFullyFusedOperations_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
        dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y,
        N_CELLS, write_rho, write_u_x, write_u_y);

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    // debugging helper
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        SPDLOG_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));
    }
}
