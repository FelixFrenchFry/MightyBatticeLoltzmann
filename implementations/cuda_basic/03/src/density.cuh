#pragma once
#include <cstdint>



void Launch_DensityFieldComputation(
    const float* const* dvc_df,
    float* rho,
    const uint32_t N_CELLS);
