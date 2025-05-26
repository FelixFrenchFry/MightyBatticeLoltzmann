#pragma once
#include "config.cuh"
#include <cstdint>



void Launch_FullyFusedOperationsComputation(
    const DF_Vec* dvc_df,
    DF_Vec* dvc_df_next,
    float* rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS);
