#pragma once
#include <cstddef>



void Launch_DensityFieldComputation(
    const float* const* dvc_df,
    float* rho,
    const size_t N_CELLS);
