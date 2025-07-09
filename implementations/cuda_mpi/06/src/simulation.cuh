#pragma once
#include "../../tools/config.cuh"
#include <cstdint>



void Launch_FullyFusedLatticeUpdate_Push_Inner(
    const FP* const* df,
    FP* const* df_new,
    FP* rho,
    FP* u_x,
    FP* u_y,
    const FP omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS_INNER,
    const int RANK,
    const bool is_SWD = false,
    const bool is_LDC = true,
    const bool write_rho = false,
    const bool write_u_x = false,
    const bool write_u_y = false);

void Launch_FullyFusedLatticeUpdate_Push_Outer(
    const FP* const* df,
    FP* const* df_new,
    FP* const* df_halo_top,
    FP* const* df_halo_bottom,
    FP* rho,
    FP* u_x,
    FP* u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_X_TOTAL, const uint32_t N_Y_TOTAL,
    const uint32_t Y_START,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS_OUTER,
    const int RANK,
    const bool is_SWD = false,
    const bool is_LDC = true,
    const bool write_rho = false,
    const bool write_u_x = false,
    const bool write_u_y = false);
