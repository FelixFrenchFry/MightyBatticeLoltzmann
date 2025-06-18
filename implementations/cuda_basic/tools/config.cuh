#pragma once
#include <cstdint>

// precision selection at compile-time
#ifdef USE_FP64
    #define FP double
    #define FP_CONST(x) x
    #define FP_SQRT sqrt
    #define FP_SIN sin
    #define FP_PI 3.14159265358979323846
#else
    #define FP float
    #define FP_CONST(x) x##f
    #define FP_SQRT sqrtf
    #define FP_SIN sinf
    #define FP_PI 3.14159265358979323846f
#endif



constexpr uint32_t N_DIR =          9;
constexpr uint32_t N_BLOCKSIZE =    256;
