#pragma once
#include <cstdint>
#include <cuda_runtime.h>


constexpr uint32_t N_DIR =          9;
constexpr uint32_t N_BLOCKSIZE =    256;
constexpr uint32_t N_VECSIZE =      4;

// aligned struct for AoSoA distribution function layout for float4 access
struct alignas(16) DF_Vec
{
    float4 df_0_to_3;   // df[0], df[1], df[2], df[3]
    float4 df_4_to_7;   // df[4], df[5], df[6], df[7]
    float  df_8;        // df[8] (accessed normally)
    float  pad[3];      // padding for alignment to 12 bytes
};
static_assert(sizeof(DF_Vec) == 48);
