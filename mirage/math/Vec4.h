#pragma once

#include "mirage/math/Vec3.h"

namespace Mirage
{
	struct Vec4
	{
		CUDA_CALLABLE inline Vec4() : x(0.0), y(0.0), z(0.0), w(0.0) { Validate(); }
		CUDA_CALLABLE inline Vec4(Real x) : x(x), y(x), z(x), w(0.0) { Validate(); }
		CUDA_CALLABLE inline Vec4(Real x, Real y, Real z, Real w = 0.0f) : x(x), y(y), z(z), w(w) { Validate(); }
		CUDA_CALLABLE inline Vec4(const Vec3 &v, Real w) : x(v.x), y(v.y), z(v.z), w(w) { Validate(); }

		CUDA_CALLABLE inline Real operator[](int index) const { return (&x)[index]; }
		CUDA_CALLABLE inline Real &operator[](int index) { return (&x)[index]; }

		CUDA_CALLABLE inline void Validate()
		{
			// assert(isfinite(x));
			// assert(isfinite(y));
			// assert(isfinite(z));
			// assert(isfinite(w));
		}

		Real x;
		Real y;
		Real z;
		Real w;
	};

	CUDA_CALLABLE inline Vec4 operator-(const Vec4 &a) { return Vec4(-a.x, -a.y, -a.z, -a.w); }
	CUDA_CALLABLE inline Vec4 operator+(const Vec4 &a, const Vec4 &b) { return Vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
	CUDA_CALLABLE inline Vec4 operator-(const Vec4 &a, const Vec4 &b) { return Vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
	CUDA_CALLABLE inline Vec4 operator*(const Vec4 &a, const Vec4 &b) { return Vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
	CUDA_CALLABLE inline Vec4 operator/(const Vec4 &a, const Vec4 &b) { return Vec4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
	CUDA_CALLABLE inline Vec4 operator*(const Vec4 &a, Real s) { return Vec4(a.x * s, a.y * s, a.z * s, a.w * s); }
	CUDA_CALLABLE inline Vec4 operator*(Real s, const Vec4 &a) { return a * s; }
	CUDA_CALLABLE inline Vec4 operator/(const Vec4 &a, Real s) { return a * (1.0 / s); }

	CUDA_CALLABLE inline Vec4 &operator+=(Vec4 &a, const Vec4 &b) { return a = a + b; }
	CUDA_CALLABLE inline Vec4 &operator-=(Vec4 &a, const Vec4 &b) { return a = a - b; }
	CUDA_CALLABLE inline Vec4 &operator*=(Vec4 &a, const Vec4 &s)
	{
		a.x *= s.x;
		a.y *= s.y;
		a.z *= s.z;
		a.w *= s.w;
		return a;
	}
	CUDA_CALLABLE inline Vec4 &operator/=(Vec4 &a, const Vec4 &s)
	{
		a.x /= s.x;
		a.y /= s.y;
		a.z /= s.z;
		a.w /= s.w;
		return a;
	}
	CUDA_CALLABLE inline Vec4 &operator*=(Vec4 &a, Real s)
	{
		a.x *= s;
		a.y *= s;
		a.z *= s;
		a.w *= s;
		return a;
	}
	CUDA_CALLABLE inline Vec4 &operator/=(Vec4 &a, Real s)
	{
		Real rcp = 1.0 / s;
		a.x *= rcp;
		a.y *= rcp;
		a.z *= rcp;
		a.w *= rcp;
		return a;
	}

	CUDA_CALLABLE inline Vec4 Cross(const Vec4 &a, const Vec4 &b);
	CUDA_CALLABLE inline Real Dot(const Vec4 &a, const Vec4 &b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
	CUDA_CALLABLE inline Real LengthSq(const Vec4 &a) { return Dot(a, a); }
	CUDA_CALLABLE inline Real Length(const Vec4 &a) { return sqrt(LengthSq(a)); }
	CUDA_CALLABLE inline Vec4 Normalize(const Vec4 &a) { return a / Length(a); }
	CUDA_CALLABLE inline Vec4 SafeNormalize(const Vec4 &a, const Vec4 &fallback = Vec4(0.0));

	CUDA_CALLABLE inline Vec4 Max(const Vec4 &a, const Vec4 &b)
	{
		return Vec4(::Mirage::Max(a.x, b.x), ::Mirage::Max(a.y, b.y), ::Mirage::Max(a.z, b.z), ::Mirage::Max(a.w, b.w));
	}

	CUDA_CALLABLE inline Vec4 Min(const Vec4 &a, const Vec4 &b)
	{
		return Vec4(::Mirage::Min(a.x, b.x), ::Mirage::Min(a.y, b.y), ::Mirage::Min(a.z, b.z), ::Mirage::Min(a.w, b.w));
	}

	CUDA_CALLABLE inline Vec3::Vec3(const Vec4 &v) : x(v.x), y(v.y), z(v.z) {}

	CUDA_CALLABLE inline Vec4 fetchVec4(const Vec4 *RESTRICT ptr, int index)
	{
#if __CUDA_ARCH__ && USE_TEXTURES
		float4 x = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index);
		return Vec4(x.x, x.y, x.z, x.w);
#else
		return ptr[index];
#endif
	}
}