#pragma once
#include <cstdint>



void Launch_ApplyShearWaveCondition_K(
    DF_Vec* dvc_df_1_to_8,
    float* dvc_df_0,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const float u_max,
    const float k,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS);
