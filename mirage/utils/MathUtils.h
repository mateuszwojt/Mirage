#pragma once

#include <cmath>

#include <cassert>
#include <algorithm>
#include <limits>
#include <random>

#include <cstdint>
#include <cstring>
#include <cfloat>

#include "mirage/utils/API.h"

// this header defines C++ types and helpers for manipulating basic vector types
// matrices are stored in column major order, with column vectors (OpenGL style)

#define kPi (3.141592653589793f)
#define k2Pi (3.141592653589793f * 2.0f)
#define kInvPi (1.0f / kPi)
#define kInv2Pi (1.0f / k2Pi)
#define kPiOver2 (kPi / 2.0f)

namespace Mirage
{
	CUDA_CALLABLE inline Real DegToRad(Real t) { return t * (kPi / 180.0f); }
	CUDA_CALLABLE inline Real RadToDeg(Real t) { return t * (180.0f / kPi); }

	CUDA_CALLABLE inline Real Sqr(Real x) { return x * x; }
	CUDA_CALLABLE inline Real Cube(Real x) { return x * x * x; }

	CUDA_CALLABLE inline Real Sign(Real x) { return x < 0.0f ? -1.0f : 1.0f; }

	CUDA_CALLABLE inline Real Sqrt(Real x) { return sqrt(x); }

	template <typename T>
	CUDA_CALLABLE inline void Swap(T &a, T &b)
	{
		T temp = a;
		a = b;
		b = temp;
	}

	template <typename T>
	CUDA_CALLABLE inline T Min(T a, T b) { return (a < b) ? a : b; }

	template <typename T>
	CUDA_CALLABLE inline T Max(T a, T b) { return (a < b) ? b : a; }

	template <typename T>
	CUDA_CALLABLE inline T Clamp(T x, T lower, T upper)
	{
		return Min(Max(x, lower), upper);
	}

	template <typename T>
	CUDA_CALLABLE inline T Abs(T x)
	{
		if (x < 0.0)
			return -x;
		else
			return x;
	}

	CUDA_CALLABLE inline float Abs(float a) { return fabsf(a); }

	template <typename T>
	CUDA_CALLABLE inline T Lerp(T a, T b, Real t)
	{
		return a + (b - a) * t;
	}

	// generic size matrix multiply. a is M x N, b is N x K, result is M x K -
	// all three stored column-major (matching Mat44/Mat33/Mat22's "cols[c][r]"
	// layout, i.e. element (row, col) lives at flat index col*rows + row), NOT
	// row-major. Indexing this as a[i*N+k]/b[k*K+j] (row-major) reads columns
	// as if they were rows, silently computing a transposed/garbled product -
	// e.g. CameraSampler's rasterToWorld (built via chained Mat44 * Mat44)
	// came out subtly wrong, skewing every generated ray off the camera's
	// true forward direction.
	template <int M, int N, int K>
	CUDA_CALLABLE inline void MatrixMultiply(Real *result, const Real *a, const Real *b)
	{
		for (int j = 0; j < K; ++j)
		{
			for (int i = 0; i < M; ++i)
			{
				Real sum = 0.0f;
				for (int k = 0; k < N; ++k)
					sum += a[k*M + i] * b[j*N + k];
				result[j*M + i] = sum;
			}
		}
	}

	// generic size matrix transpose, result must not alias a
	template <int m, int n>
	CUDA_CALLABLE inline void MatrixTranspose(Real *result, const Real *a)
	{
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				result[j + i * n] = a[i + j * m];
			}
		}
	}

	template <int m, int n>
	CUDA_CALLABLE inline void MatrixAdd(Real *result, const Real *a, const Real *b)
	{
		for (int j = 0; j < n; ++j)
		{
			for (int i = 0; i < m; ++i)
			{
				const int idx = j * m + i;
				result[idx] = a[idx] + b[idx];
			}
		}
	}

	template <int m, int n>
	CUDA_CALLABLE inline void MatrixSub(Real *result, const Real *a, const Real *b)
	{
		for (int j = 0; j < n; ++j)
		{
			for (int i = 0; i < m; ++i)
			{
				const int idx = j * m + i;
				result[idx] = a[idx] - b[idx];
			}
		}
	}

	template <int m, int n>
	CUDA_CALLABLE inline void MatrixScale(Real *result, const Real *a, const Real s)
	{
		for (int j = 0; j < n; ++j)
		{
			for (int i = 0; i < m; ++i)
			{
				const int idx = j * m + i;
				result[idx] = a[idx] * s;
			}
		}
	}

	// -----------------
	// random numbers

	class Random
	{
	public:
		CUDA_CALLABLE inline Random(int seed = 0)
		{
			seed1 = 315645664 + seed;
			seed2 = seed1 ^ 0x13ab45fe;
		}

		CUDA_CALLABLE inline unsigned int Rand()
		{
			seed1 = (seed2 ^ ((seed1 << 5) | (seed1 >> 27))) ^ (seed1 * seed2);
			seed2 = seed1 ^ ((seed2 << 12) | (seed2 >> 20));

			return seed1;
		}

		// returns a random number in the range [min, max)
		CUDA_CALLABLE inline unsigned int Rand(unsigned int min, unsigned int max)
		{
			return min + Rand() % (max - min);
		}

		// std::default_random_engine generator;

		// returns random number between 0-1
		CUDA_CALLABLE inline float Randf()
		{
			// std::uniform_real_distribution<float> distr(0.0f,1.0f);

			// return distr(generator);

			unsigned int value = Rand();
			unsigned int limit = 0xffffffff;

			return (float)value * (1.0f / (float)limit);
		}

		// returns random number between min and max
		CUDA_CALLABLE inline float Randf(float min, float max)
		{
			float t = Randf();
			return (1.0f - t) * min + t * max;
		}

		// returns random number between 0-max
		CUDA_CALLABLE inline float Randf(float max)
		{
			return Randf() * max;
		}

		unsigned int seed1;
		unsigned int seed2;
	};

	//----------------------
	// bitwise operations

	CUDA_CALLABLE inline int Part1By1(int n)
	{
		n = (n ^ (n << 8)) & 0x00ff00ff;
		n = (n ^ (n << 4)) & 0x0f0f0f0f;
		n = (n ^ (n << 2)) & 0x33333333;
		n = (n ^ (n << 1)) & 0x55555555;

		return n;
	}

	// Takes values in the range [0,1] and assigns an index based on 16bit Morton codes
	CUDA_CALLABLE inline int Morton2(Real x, Real y)
	{
		int ux = Clamp(int(x * 1024), 0, 1023);
		int uy = Clamp(int(y * 1024), 0, 1023);

		return (Part1By1(uy) << 1) + Part1By1(ux);
	}

	CUDA_CALLABLE inline int Part1by2(int n)
	{
		n = (n ^ (n << 16)) & 0xff0000ff;
		n = (n ^ (n << 8)) & 0x0300f00f;
		n = (n ^ (n << 4)) & 0x030c30c3;
		n = (n ^ (n << 2)) & 0x09249249;

		return n;
	}

	// Takes values in the range [0, 1] and assigns an index based on 10bit Morton codes
	CUDA_CALLABLE inline int Morton3(Real x, Real y, Real z)
	{
		int ux = Clamp(int(x * 1024), 0, 1023);
		int uy = Clamp(int(y * 1024), 0, 1023);
		int uz = Clamp(int(z * 1024), 0, 1023);

		return (Part1by2(uz) << 2) | (Part1by2(uy) << 1) | Part1by2(ux);
	}

	// count number of leading zeros in a 32bit word
	CUDA_CALLABLE inline int CLZ(int x)
	{
		int n;
		if (x == 0)
			return 32;
		for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1)
			;
		return n;
	}

	template <typename T>
	CUDA_CALLABLE inline T ClampLength(const T &v, float maxLength)
	{
		float l = Length(v);
		if (l > maxLength)
		{
			return v * (maxLength / l);
		}
		else
		{
			return v;
		}
	}

	CUDA_CALLABLE inline int fetchInt(const int *RESTRICT ptr, int index)
	{
#if __CUDA_ARCH__ && USE_TEXTURES
		return tex1Dfetch<int>((cudaTextureObject_t)ptr, index);
#else
		return ptr[index];
#endif
	}

	CUDA_CALLABLE inline float fetchFloat(const float *RESTRICT ptr, int index)
	{
#if __CUDA_ARCH__ && USE_TEXTURES
		return tex1Dfetch<float>((cudaTextureObject_t)ptr, index);
#else
		return ptr[index];
#endif
	}
}