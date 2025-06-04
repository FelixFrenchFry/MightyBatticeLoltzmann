#pragma once
#include <cstdint>



void Launch_ApplyShearWaveCondition_K(
    double* const* dvc_df,
    double* dvc_rho,
    double* dvc_u_x,
    double* dvc_u_y,
    const double rho_0,
    const double u_max,
    const double k,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS);

void Launch_ApplyLidDrivenCavityCondition_K(
    double* const* dvc_df,
    double* dvc_rho,
    double* dvc_u_x,
    double* dvc_u_y,
    const double rho_0,
    const uint32_t N_CELLS);
