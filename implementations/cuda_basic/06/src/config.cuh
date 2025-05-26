#pragma once
#include <cstdint>
#include <cuda_runtime.h>



constexpr uint32_t N_DIR =          9;
constexpr uint32_t N_BLOCKSIZE =    256;
constexpr uint32_t N_VECSIZE =      4;

// aligned struct for AoSoA distribution function layout for float4 access
// (restricted to 8 out of 9 directions to avoid dummy values for alignment)
struct alignas(8) DF_Vec
{
    float4 df_1_to_4;   // df[1,...,4]
    float4 df_5_to_8;   // df[5,...,8]
};
static_assert(sizeof(DF_Vec) == 32);
