#pragma once
#include "../../tools/config.cuh"
#include <cstdint>



void Launch_ApplyShearWaveCondition_K(
    FP* const* dvc_df,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP rho_0,
    const FP u_max,
    const FP k,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS);

void Launch_ApplyLidDrivenCavityCondition_K(
    FP* const* dvc_df,
    FP* dvc_rho,
    FP* dvc_u_x,
    FP* dvc_u_y,
    const FP rho_0,
    const uint32_t N_CELLS);
