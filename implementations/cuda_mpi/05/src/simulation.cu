#include "../../tools/config.cuh"
#include "../../tools/utilities.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load opposite direction vectors for bounce-back, reversed direction mapping vectors
// for halo arrays, velocity direction vectors, and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int dvc_opp_dir[9];
__constant__ int dvc_rev_dir_map_halo_top[9];
__constant__ int dvc_rev_dir_map_halo_bottom[9];
__constant__ int dvc_rev_dir_map_halo_left[9];
__constant__ int dvc_rev_dir_map_halo_right[9];
__constant__ int dvc_c_x[9];
__constant__ int dvc_c_y[9];
__constant__ FP dvc_fp_c_x[9];
__constant__ FP dvc_fp_c_y[9];
__constant__ FP dvc_w[9];
bool constantsInitialized = false;
bool kernelAttributesDisplayed_inner = false;
bool kernelAttributesDisplayed_outer = false;

void InitializeConstants()
{
    if (constantsInitialized) { return; }

    // ---------
    // | 6 2 5 |
    // | 3 0 1 |
    // | 7 4 8 |
    // ---------

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
    int rev_map_dir_halo_top[9] = { 42, 42, 0, 42, 42, 1, 2, 42, 42 }; // map 2, 5, 6 to 0, 1, 2
    int rev_map_dir_halo_bottom[9] = { 42, 42, 42, 42, 0, 42, 42, 1, 2 }; // map 4, 7, 8 to 0, 1, 2
    int rev_map_dir_halo_left[9] = { 42, 42, 42, 0, 42, 42, 1, 2, 42 }; // map 3, 6, 7 to 0, 1, 2
    int rev_map_dir_halo_right[9] = { 42, 0, 42, 42, 42, 1, 42, 42, 2 }; // map 1, 5, 8 to 0, 1, 2
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    FP fp_c_x[9] = { 0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0 };
    FP fp_c_y[9] = { 0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0 };
    FP w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(dvc_opp_dir, opp_dir, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_rev_dir_map_halo_top, rev_map_dir_halo_top, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_rev_dir_map_halo_bottom, rev_map_dir_halo_bottom, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_rev_dir_map_halo_left, rev_map_dir_halo_left, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_rev_dir_map_halo_right, rev_map_dir_halo_right, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(dvc_fp_c_x, fp_c_x, 9 * sizeof(FP));
    cudaMemcpyToSymbol(dvc_fp_c_y, fp_c_y, 9 * sizeof(FP));
    cudaMemcpyToSymbol(dvc_w, w, 9 * sizeof(FP));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

// =============================================================================
// bounce-back boundary sub-kernels TODO: adjust for dealing with halo cells
// =============================================================================
__device__ __forceinline__ void ComputeNeighborIndex_BounceBackBoundary_Conditional_K(
    uint32_t src_x, uint32_t src_y, uint32_t src_y_global,
    uint32_t N_X, uint32_t N_Y_TOTAL,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // check if directed into a wall
    if ((dvc_c_x[i] == -1 && src_x == 0) ||               // into left wall
        (dvc_c_x[i] ==  1 && src_x == N_X - 1) ||         // into right wall
        (dvc_c_y[i] == -1 && src_y == 0) ||               // into bottom wall
        (dvc_c_y[i] ==  1 && src_y == N_Y_TOTAL - 1))     // into top wall
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
    uint32_t src_x, uint32_t src_y, uint32_t src_y_global,
    uint32_t N_X, uint32_t N_Y_TOTAL,
    uint32_t i,
    uint32_t& dst_idx,
    uint32_t& dst_i)
{
    // TODO: reduce register usage
    // branch-less bit-wise computation of: 1 if bounce-back, else 0
    int bounce =
        ((dvc_c_x[i] == -1) & (src_x == 0)) |
        ((dvc_c_x[i] ==  1) & (src_x == N_X - 1)) |
        ((dvc_c_y[i] == -1) & (src_y == 0)) |
        ((dvc_c_y[i] ==  1) & (src_y == N_Y_TOTAL - 1));

    // branch-less computation of destination index
    dst_idx = bounce * (src_y * N_X + src_x)
            + (1 - bounce) * ((src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i]));

    // branch-less computation of destination direction
    dst_i = bounce * dvc_opp_dir[i] + (1 - bounce) * i;
}

// =============================================================================
// inject lid velocity sub-kernels TODO: adjust for dealing with halo cells
// =============================================================================
__device__ __forceinline__ void InjectLidVelocity_Conditional_K(
    uint32_t src_y,
    uint32_t N_Y,
    FP rho,
    FP omega,
    FP u_lid,
    uint32_t i,
    FP& f_new_i)
{
    // check if directed into the top wall
    if (dvc_c_y[i] == 1 && src_y == N_Y - 1)
    {
        f_new_i -= FP_CONST(6.0) * omega * rho * dvc_fp_c_x[i] * u_lid;
    }
}

__device__ __forceinline__ void InjectLidVelocity_BranchLess_K(
    uint32_t src_y_global,
    uint32_t N_Y_TOTAL,
    FP rho,
    FP omega,
    FP u_lid,
    uint32_t i,
    FP& f_new_i)
{
    // branch-less lid velocity injection via boolean mask
    int top_bounce = ((dvc_c_y[i] == 1) & (src_y_global == N_Y_TOTAL - 1));

    f_new_i -= top_bounce * FP_CONST(6.0) * dvc_w[i] * rho
             * dvc_fp_c_x[i] * u_lid;
}

// TODO: adjust for switch from 1D to 2D domain decomposition
// =============================================================================
// fully fused lattice update kernel for shear wave decay sim (inner cells only)
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_ShearWaveDecay_Push_Inner_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS_INNER,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS_INNER) { return; }

    // only process inner cells -> [1, ..., N_Y - 2] * N_X and
    // determine (x,y) coordinates among the inner cells
    uint32_t src_x = idx % N_X;
    uint32_t src_y = idx / N_X + 1; // starting from row 1, instead of 0
    idx = src_y * N_X + src_x;

    // load df values into block-wise tiles of shared shared memory
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // used for summing stuff up and computing collision
    FP rho = FP_CONST(0.0);
    FP u_x = FP_CONST(0.0);
    FP u_y = FP_CONST(0.0);

    // populate shared memory tiles and compute sums in the same loop
    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        tile_df[i][threadIdx.x] = dvc_df[i][idx];
        rho += tile_df[i][threadIdx.x];
        u_x += tile_df[i][threadIdx.x] * dvc_fp_c_x[i];
        u_y += tile_df[i][threadIdx.x] * dvc_fp_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho
                  * (FP_CONST(1.0) + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega
                   * (tile_df[i][threadIdx.x] - f_eq_i);

        // regular periodic boundary for shear wave decay without halo exchange
        // determine destination cell's index based on x/y-coordinates and dir i
        // (with respect to periodic boundary conditions)
        uint32_t dst_idx = (src_y + dvc_c_y[i]) * N_X
                         + ((src_x + dvc_c_x[i] + N_X) % N_X);

        // stream df value to the destination in dir i
        dvc_df_next[i][dst_idx] = f_new_i;
    }
}

// TODO: adjust for switch from 1D to 2D domain decomposition
// =============================================================================
// fully fused lattice update kernel for lid driven cavity sim (inner cells only)
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_LidDrivenCavity_Push_Inner_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS_INNER,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS_INNER) { return; }

    // only process inner cells (all except top and bottom row, left and right column)
    // determine (x, y) coordinates of the inner cell processed by this thread
    uint32_t src_x = idx % (N_X - 2) + 1; // start from column 1, instead of 0
    uint32_t src_y = idx / (N_X - 2) + 1; // start from row 1, instead of 0
    idx = src_y * N_X + src_x;

    // load df values into block-wise tiles of shared shared memory
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // used for summing stuff up and computing collision
    FP rho = FP_CONST(0.0);
    FP u_x = FP_CONST(0.0);
    FP u_y = FP_CONST(0.0);

    // populate shared memory tiles and compute sums in the same loop
    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        tile_df[i][threadIdx.x] = dvc_df[i][idx];
        rho += tile_df[i][threadIdx.x];
        u_x += tile_df[i][threadIdx.x] * dvc_fp_c_x[i];
        u_y += tile_df[i][threadIdx.x] * dvc_fp_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega * (tile_df[i][threadIdx.x] - f_eq_i);

        // no possibility of the streaming dest being directed into a wall
        // no possibility of the streaming dest being directed outside of the rank's domain
        // -> stream to regular neighbor in dir i
        dvc_df_next[i][(src_y + dvc_c_y[i]) * N_X + (src_x + dvc_c_x[i])] = f_new_i;
    }
}

void Launch_FullyFusedLatticeUpdate_Push_Inner(
    const FP* const* dvc_df,
    FP* const* dvc_df_next,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS_INNER,
    const int RANK,
    const bool shear_wave_decay,
    const bool lid_driven_cavity,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS_INNER + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    if (shear_wave_decay)
    {
        FFLU_ShearWaveDecay_Push_Inner_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
            N_CELLS_INNER, write_rho, write_u_x, write_u_y);
    }
    else if (lid_driven_cavity)
    {
        FFLU_LidDrivenCavity_Push_Inner_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
            N_CELLS_INNER, write_rho, write_u_x, write_u_y);
    }
    else
    {
        if (RANK == 0) { SPDLOG_ERROR("No valid simulation scenario selected"); }
    }

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed_inner)
    {
        if (shear_wave_decay)
        {
            DisplayKernelAttributes(FFLU_ShearWaveDecay_Push_Inner_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_ShearWaveDecay_Push_Inner_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y - 2, RANK);
        }
        else if (lid_driven_cavity)
        {
            DisplayKernelAttributes(FFLU_LidDrivenCavity_Push_Inner_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_LidDrivenCavity_Push_Inner_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y - 2, RANK);
        }

        kernelAttributesDisplayed_inner = true;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // specify detailed logging for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }
}

// TODO: adjust for switch from 1D to 2D domain decomposition
// =============================================================================
// fully fused lattice update kernel for shear wave decay sim (outer cells only)
// (for applying the periodic boundary conditions and populating the hallo cells)
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_ShearWaveDecay_Push_Outer_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* const* __restrict__ dvc_df_halo_top,
    FP* const* __restrict__ dvc_df_halo_bottom,
    FP* const* __restrict__ dvc_df_halo_left,
    FP* const* __restrict__ dvc_df_halo_right,
    FP* __restrict__ dvc_df_halo_corners,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS_OUTER,
    const bool IS_TOP_EDGE, const bool IS_BOTTOM_EDGE,
    const bool IS_LEFT_EDGE, const bool IS_RIGHT_EDGE,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS_OUTER) { return; }

    // only process outer cells -> [0, N_Y - 1] * N_X and
    // determine (x,y) coordinates among the outer cells
    int src_x = idx % N_X;
    int src_y = (idx / N_X == 0) ? 0 : (N_Y - 1); // map to first or last row
    idx = src_y * N_X + src_x;

    // load df values into block-wise tiles of shared shared memory
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // used for summing stuff up and computing collision
    FP rho = FP_CONST(0.0);
    FP u_x = FP_CONST(0.0);
    FP u_y = FP_CONST(0.0);

    // populate shared memory tiles and compute sums in the same loop
    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        tile_df[i][threadIdx.x] = dvc_df[i][idx];
        rho += tile_df[i][threadIdx.x];
        u_x += tile_df[i][threadIdx.x] * dvc_fp_c_x[i];
        u_y += tile_df[i][threadIdx.x] * dvc_fp_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;

    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega
                   * (tile_df[i][threadIdx.x] - f_eq_i);

        // determine x-coordinate of the streaming destination cell
        // (with respect to periodic boundary conditions and halo cells)
        int dst_x = (src_x + dvc_c_x[i] + N_X) % N_X;
        int dst_y_raw = src_y + dvc_c_y[i]; // possibly < 0

        // check if streaming destination is outside of the process domain
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        if (dst_y_raw == -1) // y-destination below domain -> stream into bottom halo
        {
            // map 4, 7, 8 to 0, 1, 2 using array index map for bottom halos
            dvc_df_halo_bottom[dvc_rev_dir_map_halo_bottom[i]][dst_x] = f_new_i;
        }
        else if (dst_y_raw == N_Y) // y-destination above domain -> stream into top halo
        {
            // map 2, 5, 6 to 0, 1, 2 using array index map for top halos
            dvc_df_halo_top[dvc_rev_dir_map_halo_top[i]][dst_x] = f_new_i;
        }
        else // within domain -> stream to regular neighbor in regular df arrays
        {
            dvc_df_next[i][dst_y_raw * N_X + dst_x] = f_new_i;
        }
    }
}

// TODO: adjust for switch from 1D to 2D domain decomposition
// =============================================================================
// fully fused lattice update kernel for lid driven cavity sim (outer cells only)
// (for applying the bounce-back boundary conditions, lid velocity, and populating the hallo cells)
// =============================================================================
template <uint32_t N_DIR, uint32_t N_BLOCKSIZE>
__global__ void FFLU_LidDrivenCavity_Push_Outer_K(
    const FP* const* __restrict__ dvc_df,
    FP* const* __restrict__ dvc_df_next,
    FP* const* __restrict__ dvc_df_halo_top,
    FP* const* __restrict__ dvc_df_halo_bottom,
    FP* const* __restrict__ dvc_df_halo_left,
    FP* const* __restrict__ dvc_df_halo_right,
    FP* __restrict__ dvc_df_halo_corners,
    FP* __restrict__ dvc_rho,
    FP* __restrict__ dvc_u_x,
    FP* __restrict__ dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t X_START, const uint32_t Y_START,
    const uint32_t N_CELLS_OUTER,
    const bool IS_TOP_EDGE, const bool IS_BOTTOM_EDGE,
    const bool IS_LEFT_EDGE, const bool IS_RIGHT_EDGE,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS_OUTER) { return; }

    // only process outer cells (top and bottom row, left and right column)
    // determine (x, y) coordinates of the outer cell processed by this thread
    // using a mapping to linear indices
    int linear_idx = dvc_outer_indices[idx];
    int src_x = linear_idx % N_X;
    int src_y = linear_idx / N_X;
    int src_x_global = src_x + X_START;
    int src_y_global = src_y + Y_START;
    idx = src_y * N_X + src_x;

    // load df values into block-wise tiles of shared shared memory
    __shared__ FP tile_df[N_DIR][N_BLOCKSIZE];

    // used for summing stuff up and computing collision
    FP rho = FP_CONST(0.0);
    FP u_x = FP_CONST(0.0);
    FP u_y = FP_CONST(0.0);

    // populate shared memory tiles and compute sums in the same loop
    // density := sum over df values in each dir i
    // velocity := sum over df values, weighted by each dir i
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        tile_df[i][threadIdx.x] = dvc_df[i][idx];
        rho += tile_df[i][threadIdx.x];
        u_x += tile_df[i][threadIdx.x] * dvc_fp_c_x[i];
        u_y += tile_df[i][threadIdx.x] * dvc_fp_c_y[i];
    }

    // exit thread to avoid division by zero or erroneous values
    if (rho <= FP_CONST(0.0)) { return; }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write back final field values only if requested
    if (write_rho) { dvc_rho[idx] = rho; }
    if (write_u_x) { dvc_u_x[idx] = u_x; }
    if (write_u_y) { dvc_u_y[idx] = u_y; }

    // pre-compute squared velocity for this thread's cell
    FP u_sq = u_x * u_x + u_y * u_y;

    // TODO: check for signed math using uint32_t -> incorrect results!!!
    for (uint32_t i = 0; i < N_DIR; i++)
    {
        // compute dot product of c_i * u and equilibrium df value for dir i
        FP cu = dvc_fp_c_x[i] * u_x + dvc_fp_c_y[i] * u_y;
        FP f_eq_i = dvc_w[i] * rho * (FP_CONST(1.0)
                  + FP_CONST(3.0) * cu
                  + FP_CONST(4.5) * cu * cu
                  - FP_CONST(1.5) * u_sq);

        // relax df towards equilibrium
        FP f_new_i = tile_df[i][threadIdx.x] - omega
                   * (tile_df[i][threadIdx.x] - f_eq_i);

        // determine coordinates and direction of the streaming destination cell
        // (with respect to bounce-back boundary conditions and halo cells)
        // check if streaming is directed into a wall (bounce-back)
        // ---------
        // | 6 2 5 |
        // | 3 0 1 |
        // | 7 4 8 |
        // ---------
        if ((dvc_c_x[i] == -1 && src_x == 0 && IS_LEFT_EDGE) ||        // into left wall
            (dvc_c_x[i] ==  1 && src_x == N_X - 1 && IS_RIGHT_EDGE) || // into right wall
            (dvc_c_y[i] == -1 && src_y == 0 && IS_BOTTOM_EDGE) ||      // into bottom wall
            (dvc_c_y[i] ==  1 && src_y == N_Y - 1 && IS_TOP_EDGE))     // into top wall
        {
            // inject lid velocity if streaming is directed into top wall
            if (dvc_c_y[i] ==  1 && src_y == N_Y - 1 && IS_TOP_EDGE)
            {
                // TODO: correct equation w.r.t. omega and dvc_w[i] ?
                f_new_i -= FP_CONST(6.0) * dvc_w[i] * rho * dvc_fp_c_x[i] * u_lid;
            }

            // same cell but opposite direction of dir i because of bounce-back
            // (definitely within the process domain -> stream into regular df arrays)
            dvc_df_next[dvc_opp_dir[i]][idx] = f_new_i;
        }
        else // (not directed into a wall, but might be outside of the process domain)
        {
            int dst_x_raw = src_x + dvc_c_x[i]; // possibly < 0
            int dst_y_raw = src_y + dvc_c_y[i]; // possibly < 0

            // check if streaming destination is outside of the process domain
            if (dst_y_raw == -1) // below domain, but no wall -> stream into bottom halo
            {
                if (dst_x_raw == -1) // into bottom left corner
                {
                    // write to third entry for corner in dir 7
                    dvc_df_halo_corners[2] = f_new_i;
                }
                else if (dst_x_raw == N_X) // into bottom right corner
                {
                    // write to forth (last) entry for corner in dir 8
                    dvc_df_halo_corners[3] = f_new_i;
                }
                else // no corner
                {
                    // map dirs 4, 7, 8 to 0, 1, 2 in the halo arrays
                    dvc_df_halo_bottom[dvc_rev_dir_map_halo_bottom[i]][dst_x_raw + 1] = f_new_i;
                }
            }
            else if (dst_y_raw == N_Y) // above domain, but no wall -> stream into top halo
            {
                if (dst_x_raw == -1) // into top left corner
                {
                    // write to second entry for corner in dir 6
                    dvc_df_halo_corners[1] = f_new_i;
                }
                else if (dst_x_raw == N_X) // into top right corner
                {
                    // write to first entry for corner in dir 5
                    dvc_df_halo_corners[0] = f_new_i;
                }
                else // no corner
                {
                    // map dirs 2, 5, 6 to 0, 1, 2 in the halo arrays
                    dvc_df_halo_top[dvc_rev_dir_map_halo_top[i]][dst_x_raw + 1] = f_new_i;
                }
            }
            else if (dst_x_raw == -1) // left of domain, but no wall -> stream into left halo
            {
                // map dirs 3, 6, 7 to 0, 1, 2 in the halo arrays
                dvc_df_halo_left[dvc_rev_dir_map_halo_left[i]][dst_y_raw] = f_new_i;
            }
            else if (dst_x_raw == N_X) // right of domain, but no wall -> stream into right halo
            {
                // map dirs 1, 5, 8 to 0, 1, 2 into the halo arrays
                dvc_df_halo_right[dvc_rev_dir_map_halo_right[i]][dst_y_raw] = f_new_i;
            }
            else // within domain -> stream to regular neighbor in regular df arrays
            {
                dvc_df_next[i][dst_y_raw * N_X + dst_x_raw] = f_new_i;
            }
        }
    }
}

void Launch_FullyFusedLatticeUpdate_Push_Outer(
    const FP* const* dvc_df,
    FP* const* dvc_df_next,
    FP* const* dvc_df_halo_top,
    FP* const* dvc_df_halo_bottom,
    FP* const* dvc_df_halo_left,
    FP* const* dvc_df_halo_right,
    FP* dvc_df_halo_corners,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t X_START, const uint32_t Y_START,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS_OUTER,
    const int RANK,
    const bool IS_TOP_EDGE, const bool IS_BOTTOM_EDGE,
    const bool IS_LEFT_EDGE, const bool IS_RIGHT_EDGE,
    const bool shear_wave_decay,
    const bool lid_driven_cavity,
    const bool write_rho,
    const bool write_u_x,
    const bool write_u_y)
{
    InitializeConstants();

    const uint32_t N_GRIDSIZE = (N_CELLS_OUTER + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    if (shear_wave_decay)
    {
        FFLU_ShearWaveDecay_Push_Outer_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_df_halo_top, dvc_df_halo_bottom,
            dvc_df_halo_left, dvc_df_halo_right, dvc_df_halo_corners,
            dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y, N_CELLS_OUTER,
            IS_TOP_EDGE, IS_BOTTOM_EDGE, IS_LEFT_EDGE, IS_RIGHT_EDGE,
            write_rho, write_u_x, write_u_y);
    }
    else if (lid_driven_cavity)
    {
        FFLU_LidDrivenCavity_Push_Outer_K<N_DIR, N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_next, dvc_df_halo_top, dvc_df_halo_bottom,
            dvc_df_halo_left, dvc_df_halo_right, dvc_df_halo_corners,
            dvc_rho, dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y, N_X_TOTAL, N_Y_TOTAL,
            X_START, Y_START, N_CELLS_OUTER, IS_TOP_EDGE, IS_BOTTOM_EDGE,
            IS_LEFT_EDGE, IS_RIGHT_EDGE, write_rho, write_u_x, write_u_y);
    }
    else
    {
        if (RANK == 0) { SPDLOG_ERROR("No valid simulation scenario selected"); }
    }

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed_outer)
    {
        if (shear_wave_decay)
        {
            DisplayKernelAttributes(FFLU_ShearWaveDecay_Push_Outer_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_ShearWaveDecay_Push_Outer_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, 2, RANK);
        }
        else if (lid_driven_cavity)
        {
            DisplayKernelAttributes(FFLU_LidDrivenCavity_Push_Outer_K<N_DIR, N_BLOCKSIZE>,
                fmt::format("FFLU_LidDrivenCavity_Push_Outer_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, 2, RANK);
        }

        kernelAttributesDisplayed_outer = true;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // specify detailed logging for the error message
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%s:%#] [%^%l%$] %v");

        SPDLOG_ERROR("CUDA error: {}\n", cudaGetErrorString(err));

        // return to basic logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }
}
