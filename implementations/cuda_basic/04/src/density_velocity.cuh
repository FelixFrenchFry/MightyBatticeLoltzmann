#pragma once
#include <cstdint>



void Launch_DensityAndVelocityFieldComputation(
    const float* const* dvc_df,
    float* rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const uint32_t N_CELLS);
