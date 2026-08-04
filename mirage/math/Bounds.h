#pragma once

#include "mirage/math/Vec3.h"
#include "mirage/math/Transform.h"
#include "mirage/utils/MathUtils.h"

namespace Mirage
{
	struct Bounds
	{
		CUDA_CALLABLE inline Bounds()
			: lower(REAL_MAX), upper(-REAL_MAX) {}

		CUDA_CALLABLE inline Bounds(const Vec3 &lower, const Vec3 &upper) : lower(lower), upper(upper) {}

		CUDA_CALLABLE inline Vec3 GetCenter() const { return 0.5 * (lower + upper); }
		CUDA_CALLABLE inline Vec3 GetEdges() const { return upper - lower; }

		CUDA_CALLABLE inline void Expand(Real r)
		{
			lower -= Vec3(r);
			upper += Vec3(r);
		}

		CUDA_CALLABLE inline bool Empty() const { return lower.x >= upper.x || lower.y >= upper.y; }

		CUDA_CALLABLE inline void AddPoint(const Vec3 &p)
		{
			lower = Min(lower, p);
			upper = Max(upper, p);
		}

		CUDA_CALLABLE inline bool Overlaps(const Vec3 &p) const
		{
			if (p.x < lower.x ||
				p.y < lower.y ||
				p.z < lower.z ||
				p.x > upper.x ||
				p.y > upper.y ||
				p.z > upper.z)
			{
				return false;
			}
			else
			{
				return true;
			}
		}

		CUDA_CALLABLE inline bool Overlaps(const Bounds &b) const
		{
			if (lower.x > b.upper.x ||
				lower.y > b.upper.y ||
				lower.z > b.upper.z ||
				upper.x < b.lower.x ||
				upper.y < b.lower.y ||
				upper.z < b.lower.z)
			{
				return false;
			}
			else
			{
				return true;
			}
		}

		Vec3 lower;
		Vec3 upper;
	};

	CUDA_CALLABLE inline Bounds TransformBounds(const Transform &xform, const Bounds &bounds)
	{
		// take the sum of the +/- abs value along each cartesian axis
		Mat33 m = Mat33(xform.r);

		Vec3 halfEdgeWidth = xform.s * bounds.GetEdges() * 0.5f;

		Vec3 x = Abs(m.GetCol(0)) * halfEdgeWidth.x;
		Vec3 y = Abs(m.GetCol(1)) * halfEdgeWidth.y;
		Vec3 z = Abs(m.GetCol(2)) * halfEdgeWidth.z;

		Vec3 center = TransformPoint(xform, bounds.GetCenter());

		Vec3 lower = center - x - y - z;
		Vec3 upper = center + x + y + z;

		return Bounds(lower, upper);
	}

	CUDA_CALLABLE inline Bounds Union(const Bounds &a, const Bounds &b)
	{
		return Bounds(Min(a.lower, b.lower), Max(a.upper, b.upper));
	}

	CUDA_CALLABLE inline Bounds Intersection(const Bounds &a, const Bounds &b)
	{
		return Bounds(Max(a.lower, b.lower), Min(a.upper, b.upper));
	}
}