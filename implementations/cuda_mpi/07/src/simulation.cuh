#pragma once



void Launch_FullyFusedLatticeUpdate_Push_Inner(
    const float* const* dvc_df,
    float* const* dvc_df_new,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const int N_X, const int N_Y,
    const int N_X_TOTAL, const int N_Y_TOTAL,
    const int N_STEPS,
    const int N_CELLS_INNER,
    const int RANK,
    const bool is_SWD = false,
    const bool is_LDC = true,
    const bool save_rho = false,
    const bool save_u_x = false,
    const bool save_u_y = false);

void Launch_FullyFusedLatticeUpdate_Push_Outer(
    const float* const* dvc_df,
    float* const* dvc_df_new,
    float* const* dvc_df_halo_top,
    float* const* dvc_df_halo_bot,
    float* dvc_rho,
    float* dvc_u_x,
    float* dvc_u_y,
    const float omega,
    const float u_lid,
    const int N_X, const int N_Y,
    const int N_X_TOTAL, const int N_Y_TOTAL,
    const int Y_START,
    const int N_STEPS,
    const int N_CELLS_OUTER,
    const int RANK,
    const bool is_SWD = false,
    const bool is_LDC = true,
    const bool save_rho = false,
    const bool save_u_x = false,
    const bool save_u_y = false);
