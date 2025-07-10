#pragma once



void Launch_ApplyInitialCondition_ShearWaveDecay_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const float u_max,
    const float w_num,
    const int N_X, const int N_Y,
    const int Y_START,
    const int N_CELLS);

void Launch_ApplyInitialCondition_LidDrivenCavity_K(
    float* const* dvc_df,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float rho_0,
    const int N_CELLS);
