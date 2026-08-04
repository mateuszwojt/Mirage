#pragma once

#include "mirage/math/Vec3.h"

namespace Mirage
{
    using ::Mirage::Vec2;
    using ::Mirage::Vec3;
    using ::Mirage::Random;
	CUDA_CALLABLE inline void ProjectPointToLine(::Mirage::Vec2 p, ::Mirage::Vec2 a, ::Mirage::Vec2 b, Real &t)
	{
	}

	CUDA_CALLABLE inline ::Mirage::Vec2 ClosestPointToLineSegment(::Mirage::Vec2 p, ::Mirage::Vec2 a, ::Mirage::Vec2 b, Real &t)
	{
		::Mirage::Vec2 edge = b - a;
		Real edgeLengthSq = ::Mirage::LengthSq(edge);

		if (edgeLengthSq == 0.0)
		{
			// degenerate edge, return first vertex
			t = 0.0;
			return a;
		}

		::Mirage::Vec2 delta = p - a;
		t = Dot(delta, edge) / edgeLengthSq;

		if (t <= 0.0)
		{
			return a;
		}
		else if (t >= 1.0)
		{
			return b;
		}
		else
		{
			return a + t * (b - a);
		}
	}

	// generates a transform matrix with v as the z axis, taken from PBRT
	CUDA_CALLABLE inline void BasisFromVector(const ::Mirage::Vec3 &w, ::Mirage::Vec3 *u, ::Mirage::Vec3 *v)
	{
		if (fabs(w.x) > fabs(w.y))
		{
			Real invLen = 1.0 / sqrt(w.x * w.x + w.z * w.z);
			*u = ::Mirage::Vec3(-w.z * invLen, 0.0f, w.x * invLen);
		}
		else
		{
			Real invLen = 1.0 / sqrt(w.y * w.y + w.z * w.z);
			*u = ::Mirage::Vec3(0.0f, w.z * invLen, -w.y * invLen);
		}

		*v = Cross(w, *u);
	}

	CUDA_CALLABLE inline ::Mirage::Vec3 UniformSampleSphere(float u1, float u2)
	{
		float z = 1.f - 2.f * u1;
		float r = sqrtf(::Mirage::Max(0.f, 1.f - z * z));
		float phi = 2.f * kPi * u2;
		float x = r * cosf(phi);
		float y = r * sinf(phi);

		return ::Mirage::Vec3(x, y, z);
	}

	CUDA_CALLABLE inline ::Mirage::Vec3 UniformSampleHemisphere(::Mirage::Random &rand)
	{
		// generate a random z value
		float z = rand.Randf(0.0f, 1.0f);
		float w = sqrt(1.0f - z * z);

		float phi = k2Pi * rand.Randf(0.0f, 1.0f);
		float x = cosf(phi) * w;
		float y = sinf(phi) * w;

		return ::Mirage::Vec3(x, y, z);
	}

	CUDA_CALLABLE inline Vec2 UniformSampleDisc(float u1, float u2)
	{
		float r = sqrt(u1);
		float theta = k2Pi * u2;

		return ::Mirage::Vec2(r * cos(theta), r * sin(theta));
	}

	CUDA_CALLABLE inline void UniformSampleTriangle(::Mirage::Random &rand, float &u, float &v)
	{
		float r = sqrt(rand.Randf());
		u = 1.0f - r;
		v = rand.Randf() * r;
	}

	CUDA_CALLABLE inline ::Mirage::Vec3 CosineSampleHemisphere(float u1, float u2)
	{
		Vec2 s = UniformSampleDisc(u1, u2);
		float z = sqrt(::Mirage::Max(0.0f, 1.0f - s.x * s.x - s.y * s.y));

		return ::Mirage::Vec3(s.x, s.y, z);
	}

	CUDA_CALLABLE inline ::Mirage::Vec3 SphericalToXYZ(float theta, float phi)
	{
		float cosTheta = cos(theta);
		float sinTheta = sin(theta);

		return ::Mirage::Vec3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(theta) * cos(phi));
	}

	// make v lie in the same hemisphere as n
	CUDA_CALLABLE inline ::Mirage::Vec3 FaceForward(const ::Mirage::Vec3 &n, const ::Mirage::Vec3 &v)
	{
		if (Dot(v, n) < 0.0f)
			return -n;
		else
			return n;
	}
} // namespace Mirage
