#pragma once
#include <cstddef>



void Launch_VelocityFieldComputation(
    const float* const* dvc_df,
    const float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const size_t N_CELLS);
