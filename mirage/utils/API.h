#pragma once

#define USE_DOUBLE_PRECISION 0

#if USE_DOUBLE_PRECISION
typedef double Real;
#define REAL_MAX DBL_MAX
#else
typedef float Real;
#define REAL_MAX FLT_MAX
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#define USE_TEXTURES 0

#if __CUDA_ARCH__
#define RESTRICT __restrict__
#else
#define RESTRICT 
#endif