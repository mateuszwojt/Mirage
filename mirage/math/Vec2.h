#pragma once

#include "mirage/utils/MathUtils.h"

namespace Mirage
{
	struct Vec2
	{
		CUDA_CALLABLE inline Vec2() : x(0.0f), y(0.0f) {}
		CUDA_CALLABLE inline Vec2(Real x) : x(x), y(x) {}
		CUDA_CALLABLE inline Vec2(Real x, Real y) : x(x), y(y) {}

		CUDA_CALLABLE inline Real operator[](int index) const
		{
			assert(index < 2);
			return (&x)[index];
		}
		CUDA_CALLABLE inline Real &operator[](int index)
		{
			assert(index < 2);
			return (&x)[index];
		}

		Real x;
		Real y;
	};

	CUDA_CALLABLE inline Vec2 Max(const Vec2 &a, const Vec2 &b)
	{
		return Vec2(Max(a.x, b.x), Max(a.y, b.y));
	}

	CUDA_CALLABLE inline Vec2 Min(const Vec2 &a, const Vec2 &b)
	{
		return Vec2(Min(a.x, b.x), Min(a.y, b.y));
	}

	CUDA_CALLABLE inline Vec2 operator-(const Vec2 &a) { return Vec2(-a.x, -a.y); }
	CUDA_CALLABLE inline Vec2 operator+(const Vec2 &a, const Vec2 &b) { return Vec2(a.x + b.x, a.y + b.y); }
	CUDA_CALLABLE inline Vec2 operator-(const Vec2 &a, const Vec2 &b) { return Vec2(a.x - b.x, a.y - b.y); }
	CUDA_CALLABLE inline Vec2 operator*(const Vec2 &a, Real s) { return Vec2(a.x * s, a.y * s); }
	CUDA_CALLABLE inline Vec2 operator*(Real s, const Vec2 &a) { return a * s; }
	CUDA_CALLABLE inline Vec2 operator*(const Vec2 &a, const Vec2 &b) { return Vec2(a.x * b.x, a.y * b.y); }
	CUDA_CALLABLE inline Vec2 operator/(const Vec2 &a, Real s) { return a * (1.0 / s); }
	CUDA_CALLABLE inline Vec2 operator/(const Vec2 &a, const Vec2 &b) { return Vec2(a.x / b.x, a.y / b.y); }

	CUDA_CALLABLE inline Vec2 &operator+=(Vec2 &a, const Vec2 &b) { return a = a + b; }
	CUDA_CALLABLE inline Vec2 &operator-=(Vec2 &a, const Vec2 &b) { return a = a - b; }
	CUDA_CALLABLE inline Vec2 &operator*=(Vec2 &a, Real s)
	{
		a.x *= s;
		a.y *= s;
		return a;
	}
	CUDA_CALLABLE inline Vec2 &operator*=(Vec2 &a, const Vec2 &b)
	{
		a.x *= b.x;
		a.y *= b.y;
		return a;
	}
	CUDA_CALLABLE inline Vec2 &operator/=(Vec2 &a, Real s)
	{
		Real rcp = 1.0 / s;
		a.x *= rcp;
		a.y *= rcp;
		return a;
	}
	CUDA_CALLABLE inline Vec2 &operator/=(Vec2 &a, const Vec2 &b)
	{
		a.x /= b.x;
		a.y /= b.y;
		return a;
	}

	CUDA_CALLABLE inline Vec2 PerpCCW(const Vec2 &v) { return Vec2(-v.y, v.x); }
	CUDA_CALLABLE inline Vec2 PerpCW(const Vec2 &v) { return Vec2(v.y, -v.x); }
	CUDA_CALLABLE inline Real Dot(const Vec2 &a, const Vec2 &b) { return a.x * b.x + a.y * b.y; }
	CUDA_CALLABLE inline Real LengthSq(const Vec2 &a) { return Dot(a, a); }
	CUDA_CALLABLE inline Real Length(const Vec2 &a) { return sqrt(LengthSq(a)); }
	CUDA_CALLABLE inline Vec2 Normalize(const Vec2 &a) { return a / Length(a); }
	CUDA_CALLABLE inline Vec2 SafeNormalize(const Vec2 &a, const Vec2 &fallback = Vec2(0.0))
	{
		Real l = Length(a);
		if (l > 0.0)
			return a / l;
		else
			return fallback;
	}

	CUDA_CALLABLE inline Vec2 fetchVec2(const Vec2 *RESTRICT ptr, int index)
	{
#if __CUDA_ARCH__ && USE_TEXTURES

		float4 x = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index);
		return Vec3(x.x, x.y, x.z);
#else
		return ptr[index];
#endif
	}
}