#pragma once
#include <cstdint>



void Launch_CollisionComputation(
    float* const* dvc_df,
    const float* dvc_rho,
    const float* dvc_u_x,
    const float* dvc_u_y,
    const float omega,
    const uint32_t N_CELLS);
