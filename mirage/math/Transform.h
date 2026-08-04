#pragma once

#include "mirage/math/Vec3.h"
#include "mirage/math/Mat33.h"

namespace Mirage
{
	struct Transform
	{
		// transform
		CUDA_CALLABLE inline Transform() : p(0.0), s(1.0f) {}
		CUDA_CALLABLE inline Transform(const Vec3 &v, const Quat &r = Quat(), float s = 1.0f) : p(v), r(r), s(s) {}

		CUDA_CALLABLE inline Transform operator*(const Transform &rhs) const
		{
			return Transform(Rotate(r, rhs.p) + p, r * rhs.r, s * rhs.s);
		}

		Vec3 p;
		Quat r;
		float s;
	};

	CUDA_CALLABLE inline Transform Inverse(const Transform &transform)
	{
		Transform t;
		t.r = Conjugate(transform.r);
		t.p = -Rotate(t.r, transform.p);
		t.s = 1.0f / transform.s;

		return t;
	}

	CUDA_CALLABLE inline Vec3 TransformVector(const Transform &t, const Vec3 &v)
	{
		return t.r * (t.s * v);
	}

	CUDA_CALLABLE inline Vec3 TransformPoint(const Transform &t, const Vec3 &v)
	{
		return t.p + t.r * (t.s * v);
	}

	CUDA_CALLABLE inline Vec3 InverseTransformVector(const Transform &t, const Vec3 &v)
	{
		return (1.0f / t.s) * (Conjugate(t.r) * v);
	}

	CUDA_CALLABLE inline Vec3 InverseTransformPoint(const Transform &t, const Vec3 &v)
	{
		return (1.0f / t.s) * (Conjugate(t.r) * (v - t.p));
	}

	CUDA_CALLABLE inline Transform InterpolateTransform(const Transform &a, const Transform &b, float t)
	{
		return Transform(::Mirage::Lerp(a.p, b.p, t), Normalize(::Mirage::Lerp(a.r, b.r, t)), ::Mirage::Lerp(a.s, b.s, t));
	}

} // namespace Mirage
