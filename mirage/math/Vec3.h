#pragma once

#include "mirage/math/Vec2.h"

namespace Mirage
{
	struct Vec4;
	struct Vec3
	{
		CUDA_CALLABLE inline Vec3() : x(0.0), y(0.0), z(0.0) {}
		// CUDA_CALLABLE inline Vec3() {}
		CUDA_CALLABLE inline Vec3(Real x) : x(x), y(x), z(x) { Validate(); }
		CUDA_CALLABLE inline Vec3(Real x, Real y, Real z) : x(x), y(y), z(z) { Validate(); }
		CUDA_CALLABLE inline Vec3(const Vec2 &v, Real z) : x(v.x), y(v.y), z(z) { Validate(); }

		CUDA_CALLABLE inline explicit Vec3(const Vec4 &v);

		CUDA_CALLABLE inline Real operator[](int index) const
		{
			assert(index < 3);
			return (&x)[index];
		}
		CUDA_CALLABLE inline Real &operator[](int index)
		{
			assert(index < 3);
			return (&x)[index];
		}

		CUDA_CALLABLE inline void Validate() const
		{
			// assert(isfinite(x) && isfinite(y) && isfinite(z));
		}

		Real x;
		Real y;
		Real z;
	};

	CUDA_CALLABLE inline Vec3 operator-(const Vec3 &a) { return Vec3(-a.x, -a.y, -a.z); }
	CUDA_CALLABLE inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
	CUDA_CALLABLE inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
	CUDA_CALLABLE inline Vec3 operator*(const Vec3 &a, Real s) { return Vec3(a.x * s, a.y * s, a.z * s); }
	CUDA_CALLABLE inline Vec3 operator*(Real s, const Vec3 &a) { return a * s; }
	CUDA_CALLABLE inline Vec3 operator*(const Vec3 &a, const Vec3 &b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
	CUDA_CALLABLE inline Vec3 operator/(const Vec3 &a, Real s) { return a * (1.0 / s); }
	CUDA_CALLABLE inline Vec3 operator/(const Vec3 &a, const Vec3 &b) { return Vec3(a.x / b.x, a.y / b.y, a.z / b.z); }

	CUDA_CALLABLE inline Vec3 &operator+=(Vec3 &a, const Vec3 &b) { return a = a + b; }
	CUDA_CALLABLE inline Vec3 &operator-=(Vec3 &a, const Vec3 &b) { return a = a - b; }
	CUDA_CALLABLE inline Vec3 &operator*=(Vec3 &a, const Vec3 &b) { return a = a * b; }
	CUDA_CALLABLE inline Vec3 &operator/=(Vec3 &a, const Vec3 &b) { return a = a / b; }

	CUDA_CALLABLE inline Vec3 &operator*=(Vec3 &a, Real s)
	{
		a.x *= s;
		a.y *= s;
		a.z *= s;
		return a;
	}
	CUDA_CALLABLE inline Vec3 &operator/=(Vec3 &a, Real s)
	{
		Real rcp = 1.0 / s;
		a.x *= rcp;
		a.y *= rcp;
		a.z *= rcp;
		return a;
	}

	CUDA_CALLABLE inline Vec3 Exp(const Vec3 &v) { return Vec3(expf(v.x), expf(v.y), expf(v.z)); }
	CUDA_CALLABLE inline Vec3 Log(const Vec3 &v) { return Vec3(logf(v.x), logf(v.y), logf(v.z)); }

	CUDA_CALLABLE inline Vec3 Cross(const Vec3 &a, const Vec3 &b) { return Vec3(a.y * b.z - b.y * a.z, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
	CUDA_CALLABLE inline Real Dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	CUDA_CALLABLE inline Real LengthSq(const Vec3 &a) { return Dot(a, a); }
	CUDA_CALLABLE inline Real Length(const Vec3 &a) { return sqrt(LengthSq(a)); }
	CUDA_CALLABLE inline Vec3 Normalize(const Vec3 &a) { return a / Length(a); }
	CUDA_CALLABLE inline Vec3 SafeNormalize(const Vec3 &a, const Vec3 &fallback = Vec3(0.0))
	{
		Real m = LengthSq(a);

		if (m > 0.0)
		{
			return a * (1.0 / sqrt(m));
		}
		else
		{
			return fallback;
		}
	}

	CUDA_CALLABLE inline Vec3 Abs(const Vec3 &a) { return Vec3(Abs(a.x), Abs(a.y), Abs(a.z)); }

	CUDA_CALLABLE inline Vec3 Max(const Vec3 &a, const Vec3 &b)
	{
		return Vec3(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z));
	}

	CUDA_CALLABLE inline Vec3 Min(const Vec3 &a, const Vec3 &b)
	{
		return Vec3(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z));
	}

	CUDA_CALLABLE inline Vec3 fetchVec3(const Vec3 *RESTRICT ptr, int index)
	{
#if __CUDA_ARCH__ && USE_TEXTURES

		float4 x = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index);
		return Vec3(x.x, x.y, x.z);
#else
		return ptr[index];
#endif
	}
}