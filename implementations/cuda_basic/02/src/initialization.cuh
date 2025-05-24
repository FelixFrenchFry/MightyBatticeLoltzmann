#pragma once
#include <cstddef>



void Launch_ApplyShearWaveCondition_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const float u_max,
    const float k,
    const size_t N_X, const size_t N_Y,
    const size_t N_CELLS);
