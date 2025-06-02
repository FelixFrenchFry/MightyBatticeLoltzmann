#pragma once
#include <cstdint>



void Launch_FullyFusedOperationsComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    float* rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool export_rho = false,
    const bool export_u_x = false,
    const bool export_u_y = false);
