#pragma once
#include <cstdint>



void Launch_StreamingComputation(
    const float* const* dvc_df,
    float* const* dvc_df_next,
    const uint32_t N_X, const uint32_t N_Y,
    const uint32_t N_CELLS);
