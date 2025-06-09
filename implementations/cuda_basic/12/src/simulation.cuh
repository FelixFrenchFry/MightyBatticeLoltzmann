#pragma once
#include <cstdint>



void Launch_FullyFusedOperationsComputation(
    const float* dvc_df_flat,
    float* dvc_df_next_flat,
    float* rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_STEPS,
    const uint32_t N_CELLS);
