#pragma once
#include <cstdint>



void Launch_FullyFusedOperationsComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const float u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool write_rho = false,
    const bool write_u_x = false,
    const bool write_u_y = false);
