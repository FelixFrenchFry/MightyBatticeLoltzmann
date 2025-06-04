#pragma once
#include <cstdint>



void Launch_FullyFusedOperationsComputation(
    const double* const* dvc_df,
    double* const* dvc_df_next,
    double* dvc_rho,
    double* dvc_u_x,
    double* dvc_u_y,
    const double omega,
    const double u_lid,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS,
    const bool write_rho = false,
    const bool write_u_x = false,
    const bool write_u_y = false);
