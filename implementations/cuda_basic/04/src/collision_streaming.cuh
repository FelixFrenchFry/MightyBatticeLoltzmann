#pragma once
#include <cstdint>



void Launch_CollisionAndStreamingComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    const float* dvc_rho,
    const float* dvc_u_x,
    const float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS);
