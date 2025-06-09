#pragma once
#include "../../tools/config.cuh"
#include <cstdint>



void Launch_FullyFusedOperationsComputation(
    const FP* const* dvc_df,
    FP* const* dvc_df_next,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP omega,
    const FP u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS,
    const bool write_rho = false,
    const bool write_u_x = false,
    const bool write_u_y = false);
