#pragma once
#include <cstddef>



void Launch_StreamingComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    const size_t N_X, const size_t N_Y,
    const size_t N_CELLS);
