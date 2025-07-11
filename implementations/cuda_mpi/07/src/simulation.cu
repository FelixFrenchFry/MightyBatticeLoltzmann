#include "../../tools/config.cuh"
#include "../../tools/utilities.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <spdlog/spdlog.h>



// load opposite direction vectors for bounce-back, reversed direction mapping vectors
// for halo arrays, velocity direction vectors, and weight vectors into constant memory
// (fast, global, read-only lookup table identical for all threads)
// TODO: figure out how to safely use same constant memory across all .cu files
__constant__ int con_opp_dir[9];
__constant__ int con_rev_dir_map_halo_top[9];
__constant__ int con_rev_dir_map_halo_bot[9];
__constant__ int con_c_x[9];
__constant__ int con_c_y[9];
__constant__ float con_w[9];
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
    int rev_map_dir_halo_bot[9] = { 42, 42, 42, 42, 0, 42, 42, 1, 2 }; // map 4, 7, 8 to 0, 1, 2
    int c_x[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
    int c_y[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
    float w[9] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                   1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };

    // copy them into constant memory on the device
    cudaMemcpyToSymbol(con_opp_dir, opp_dir, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_rev_dir_map_halo_top, rev_map_dir_halo_top, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_rev_dir_map_halo_bot, rev_map_dir_halo_bot, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_c_x, c_x, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_c_y, c_y, 9 * sizeof(int));
    cudaMemcpyToSymbol(con_w, w, 9 * sizeof(float));

    cudaDeviceSynchronize();
    constantsInitialized = true;
}

// =============================================================================
// fully fused lattice update kernel for LDC sim (inner cells only)
// =============================================================================
template <int N_BLOCKSIZE>
__global__ void FFLU_LidDrivenCavity_Push_Inner_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_new,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float omega,
    const int N_X, const int N_Y,
    const int N_CELLS_INNER,
    const bool save_rho,
    const bool save_u_x,
    const bool save_u_y)
{
    // thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS_INNER) { return; }

    // determine (x, y) coordinates of cell processed by this thread
    int cell_x = idx % N_X;
    int cell_y = idx / N_X + 1; // skip bottom row
    idx += N_X; // translate into cell index

    // temp accumulators
    float rho = 0.0f;
    float u_x = 0.0f;
    float u_y = 0.0f;

    for (int i = 0; i < 9; i++)
    {
        // fully coalesced load from global memory
        float df_val = dvc_df[i][idx];

        rho += df_val;
        u_x += df_val * con_c_x[i];
        u_y += df_val * con_c_y[i];
    }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write final field values to global device memory
    if (save_rho) { dvc_rho[idx] = rho; }
    if (save_u_x) { dvc_u_x[idx] = u_x; }
    if (save_u_y) { dvc_u_y[idx] = u_y; }

    // squared velocity
    float u_sq = u_x * u_x + u_y * u_y;

    for (int i = 0; i < 9; i++)
    {
        // dot product of c_i * u
        float cu = con_c_x[i] * u_x + con_c_y[i] * u_y;

        // equilibrium value for direction i
        float df_eq_i = con_w[i] * rho
                      * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // relax towards equilibrium
        float df_new_i = dvc_df[i][idx] - omega * (dvc_df[i][idx] - f_eq_i);

        // check if streaming into a boundary
        if ((con_c_x[i] == -1 && cell_x == 0) ||     // left boundary
            (con_c_x[i] ==  1 && cell_x == N_X - 1)) // right boundary
        {
            // bounce-back stream into same cell, but opposite direction
            dvc_df_new[con_opp_dir[i]][cell_y * N_X + cell_x] = f_new_i;
        }
        else
        {
            // destination coordiantes
            int dst_x = cell_x + con_c_x[i];
            int dst_y = cell_y + con_c_y[i];

            // stream to destination cell in direction i
            dvc_df_new[i][cell_y * N_X + cell_x] = f_new_i;
        }
    }
}

void Launch_FullyFusedLatticeUpdate_Push_Inner(
    const float* const* dvc_df,
    float* const* dvc_df_new,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const int N_X, const int N_Y,
    const int N_X_TOTAL, const int N_Y_TOTAL,
    const int N_STEPS,
    const int N_CELLS_INNER,
    const int RANK,
    const bool is_SWD,
    const bool is_LDC,
    const bool save_rho,
    const bool save_u_x,
    const bool save_u_y)
{
    InitializeConstants();

    const int N_GRIDSIZE = (N_CELLS_INNER + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    if (is_SWD)
    {
        if (RANK == 0) { SPDLOG_ERROR("Shear wave decay not availabe in this presentation-version"); }
    }
    else if (is_LDC)
    {
        FFLU_LidDrivenCavity_Push_Inner_K<N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_new, dvc_rho, dvc_u_x, dvc_u_y, omega, N_X, N_Y,
            N_CELLS_INNER, save_rho, save_u_x, save_u_y);
    }
    else
    {
        if (RANK == 0) { SPDLOG_ERROR("No valid simulation mode selected"); }
    }

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed_inner)
    {
        if (is_LDC)
        {
            DisplayKernelAttributes(FFLU_LidDrivenCavity_Push_Inner_K<N_BLOCKSIZE>,
                fmt::format("FFLU_LidDrivenCavity_Push_Inner_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, N_Y - 2, RANK);
        }

        kernelAttributesDisplayed_inner = true;
    }

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

// =============================================================================
// fully fused lattice update kernel for LDC sim (outer cells only)
// (for applying the bbbc, lid velocity, and populating the hallo cells)
// =============================================================================
template <int N_BLOCKSIZE>
__global__ void FFLU_LidDrivenCavity_Push_Outer_K(
    const float* const* __restrict__ dvc_df,
    float* const* __restrict__ dvc_df_new,
    float* const* __restrict__ dvc_df_halo_top,
    float* const* __restrict__ dvc_df_halo_bot,
    float* __restrict__ dvc_rho,
    float* __restrict__ dvc_u_x,
    float* __restrict__ dvc_u_y,
    const float omega,
    const float u_lid,
    const int N_X, const int N_Y,
    const int N_Y_TOTAL, const int Y_START,
    const int N_CELLS_OUTER,
    const bool save_rho,
    const bool save_u_x,
    const bool save_u_y)
{
    // thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS_OUTER) { return; }

    // determine (x, y) coordinates of cell processed by this thread
    int cell_x = idx % N_X;
    int cell_y = (idx / N_X == 0) ? 0 : (N_Y - 1); // map to first or last row
    int cell_y_global = cell_y + Y_START;
    idx = cell_y * N_X + cell_x; // cell index

    // temp accumulators
    float rho = 0.0f;
    float u_x = 0.0f;
    float u_y = 0.0f;

    for (int i = 0; i < 9; i++)
    {
        // fully coalesced load from global memory
        float df_val = dvc_df[i][idx];

        rho += df_val;
        u_x += df_val * con_c_x[i];
        u_y += df_val * con_c_y[i];
    }

    // finalize velocities
    u_x /= rho;
    u_y /= rho;

    // write final field values to global device memory
    if (save_rho) { dvc_rho[idx] = rho; }
    if (save_u_x) { dvc_u_x[idx] = u_x; }
    if (save_u_y) { dvc_u_y[idx] = u_y; }

    // squared velocity
    float u_sq = u_x * u_x + u_y * u_y;

    for (int i = 0; i < 9; i++)
    {
        // dot product of c_i * u
        float cu = con_c_x[i] * u_x + con_c_y[i] * u_y;

        // equilibrium value for direction i
        float df_eq_i = con_w[i] * rho
                      * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u_sq);

        // relax towards equilibrium
        float df_new_i = dvc_df[i][idx] - omega * (dvc_df[i][idx] - f_eq_i);

        // check if streaming into a boundary
        if ((con_c_x[i] == -1 && cell_x == 0) ||                  // left boundary
            (con_c_x[i] ==  1 && cell_x == N_X - 1) ||            // right boundary
            (con_c_y[i] == -1 && cell_y_global == 0) ||           // bottom boundary
            (con_c_y[i] ==  1 && cell_y_global == N_Y_TOTAL - 1)) // top boundary
        {
            // top boundary? inject lid velocity
            if (con_c_y[i] == 1 && cell_y_global == N_Y_TOTAL - 1)
            {
                f_new_i -= 6.0f * con_w[i] * rho * con_c_x[i] * u_lid;
            }

            // bounce-back stream into same cell, but opposite direction
            dvc_df_new[con_opp_dir[i]][cell_y * N_X + cell_x] = f_new_i;
        }
        else // (possibly outside of the sub-domain)
        {
            // destination coordiantes
            int dst_x = cell_x + con_c_x[i];
            int dst_y = cell_y + con_c_y[i];

            if (dst_y == -1) // below domain -> stream into bottom halo
            {
                // map 4, 7, 8 to 0, 1, 2
                dvc_df_halo_bot[con_rev_dir_map_halo_bot[i]][dst_x] = f_new_i;
            }
            else if (dst_y == N_Y) // above domain -> stream into top halo
            {
                // map 2, 5, 6 to 0, 1, 2
                dvc_df_halo_top[con_rev_dir_map_halo_top[i]][dst_x] = f_new_i;
            }
            else // within domain -> stream to destination cell in direction i
            {
                dvc_df_new[i][dst_y * N_X + dst_x] = f_new_i;
            }
        }
    }
}

void Launch_FullyFusedLatticeUpdate_Push_Outer(
    const float* const* dvc_df,
    float* const* dvc_df_new,
    float* const* dvc_df_halo_top,
    float* const* dvc_df_halo_bot,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const float u_lid,
    const int N_X, const int N_Y,
    const int N_X_TOTAL, const int N_Y_TOTAL,
    const int Y_START,
    const int N_STEPS,
    const int N_CELLS_OUTER,
    const int RANK,
    const bool is_SWD,
    const bool is_LDC,
    const bool save_rho,
    const bool save_u_x,
    const bool save_u_y)
{
    InitializeConstants();

    const int N_GRIDSIZE = (N_CELLS_OUTER + N_BLOCKSIZE - 1) / N_BLOCKSIZE;

    if (is_SWD)
    {
        if (RANK == 0) { SPDLOG_ERROR("Shear wave decay not availabe in this presentation-version"); }
    }
    else if (is_LDC)
    {
        FFLU_LidDrivenCavity_Push_Outer_K<N_BLOCKSIZE><<<N_GRIDSIZE, N_BLOCKSIZE>>>(
            dvc_df, dvc_df_new, dvc_df_halo_top, dvc_df_halo_bot, dvc_rho,
            dvc_u_x, dvc_u_y, omega, u_lid, N_X, N_Y, N_Y_TOTAL, Y_START,
            N_CELLS_OUTER, save_rho, save_u_x, save_u_y);
    }
    else
    {
        if (RANK == 0) { SPDLOG_ERROR("No valid simulation mode selected"); }
    }

    // wait for GPU to finish operations
    cudaDeviceSynchronize();

    if (!kernelAttributesDisplayed_outer)
    {
        if (is_LDC)
        {
            DisplayKernelAttributes(FFLU_LidDrivenCavity_Push_Outer_K<N_BLOCKSIZE>,
                fmt::format("FFLU_LidDrivenCavity_Push_Outer_K"),
                N_GRIDSIZE, N_BLOCKSIZE, N_X, 2, RANK);
        }

        kernelAttributesDisplayed_outer = true;
    }

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
