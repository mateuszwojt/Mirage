#pragma once

#include "mirage/core/Scene.h"
#include "mirage/core/Sampler.h"
#include "mirage/core/Probe.h"
#include "mirage/math/Vec4.h"
#include "mirage/math/Transform.h"
#include "mirage/math/Geometric.h"
#include "mirage/prims/Mesh.h"
#include "mirage/core/BVH.h"

namespace Mirage
{
    using ::Mirage::Vec2;
    using ::Mirage::Vec3;
    using ::Mirage::Vec4;
    using ::Mirage::Primitive;
    using ::Mirage::Light;
    using ::Mirage::Camera;
    using ::Mirage::MeshGeometry;
    using ::Mirage::BVHNode;
    using ::Mirage::eSphere;
    using ::Mirage::eMesh;
    using ::Mirage::ePlane;
    using ::Mirage::Bounds;
    using ::Mirage::Transform;
    using ::Mirage::Color;
    using ::Mirage::LowerBound;

	template <typename T>
	CUDA_CALLABLE CUDA_CALLABLE inline void Sort2(T &a, T &b)
	{
		if (b < a)
			::Mirage::Swap(a, b);
	}

	template <typename T>
	CUDA_CALLABLE CUDA_CALLABLE inline void Sort3(T &a, T &b, T &c)
	{
		if (b < a)
			Swap(a, b);
		if (c < b)
			Swap(b, c);
		if (b < a)
			Swap(a, b);

		assert(a <= b);
		assert(b <= c);
	}

	template <typename T>
	CUDA_CALLABLE CUDA_CALLABLE inline bool SolveQuadratic(T a, T b, T c, T &minT, T &maxT)
	{
		if (a == 0.0f && b == 0.0f)
		{
			minT = maxT = 0.0f;
			return true;
		}

		T discriminant = b * b - T(4.0) * a * c;

		if (discriminant < 0.0f)
		{
			return false;
		}

		// numerical receipes 5.6 (this method ensures numerical accuracy is preserved)
		T t = T(-0.5) * (b + ::Mirage::Sign(b) * ::Mirage::Sqrt(discriminant));
		minT = t / a;
		maxT = c / t;

		Sort2(minT, maxT);

		return true;
	}

	// alternative ray sphere intersect, returns closest and furthest t values
	CUDA_CALLABLE inline bool IntersectRaySphere(const ::Mirage::Vec3&sphereOrigin, float sphereRadius, const ::Mirage::Vec3&rayOrigin, const ::Mirage::Vec3&rayDir, float &minT, float &maxT, ::Mirage::Vec3*hitNormal = nullptr)
	{
		::Mirage::Vec3 q = rayOrigin - sphereOrigin;

		float a = 1.0f;
		float b = 2.0f * Dot(q, rayDir);
		float c = Dot(q, q) - (sphereRadius * sphereRadius);

		bool r = SolveQuadratic(a, b, c, minT, maxT);

		if (minT < 0.0f && maxT < 0.0f)
			return false;

		if (minT < 0.0f && maxT > 0.0f)
			minT = maxT;

		// calculate the normal of the closest hit
		if (hitNormal && r)
		{
			*hitNormal = Normalize((rayOrigin + rayDir * minT) - sphereOrigin);
		}

		return r;
	}

	CUDA_CALLABLE inline bool IntersectRayPlane(const ::Mirage::Vec3&p, const ::Mirage::Vec3&dir, const ::Mirage::Vec4 &plane, float &t)
	{
		float d = Dot(plane, ::Mirage::Vec4(dir, 0.0f));

		if (d == 0.0f)
		{
			return false;
		}
		else
		{
			t = -Dot(plane, ::Mirage::Vec4(p, 1.0f)) / d;
		}

		return (t > 0.0f);
	}

	CUDA_CALLABLE inline bool IntersectLineSegmentPlane(const ::Mirage::Vec3&start, const ::Mirage::Vec3&end, const ::Mirage::Vec4 &plane, ::Mirage::Vec3&out)
	{
		::Mirage::Vec3 u(end - start);

		float dist = -Dot(plane, ::Mirage::Vec4(start, 1.0f)) / Dot(plane, ::Mirage::Vec4(u, 0.0f));

		if (dist > 0.0f && dist < 1.0f)
		{
			out = (start + u * dist);
			return true;
		}
		else
			return false;
	}

	// Moller and Trumbore's method
	CUDA_CALLABLE inline bool IntersectRayTriTwoSided(const ::Mirage::Vec3&p, const ::Mirage::Vec3&dir, const ::Mirage::Vec3&a, const ::Mirage::Vec3&b, const ::Mirage::Vec3&c, float &t, float &u, float &v, float &w, float &sign, ::Mirage::Vec3*normal)
	{
		::Mirage::Vec3 ab = b - a;
		::Mirage::Vec3 ac = c - a;
		::Mirage::Vec3 n = Cross(ab, ac);

		float d = Dot(-dir, n);
		float ood = 1.0f / d; // No need to check for division by zero here as infinity aritmetic will save us...
		::Mirage::Vec3 ap = p - a;

		t = Dot(ap, n) * ood;
		if (t < 0.0f)
			return false;

		::Mirage::Vec3 e = Cross(-dir, ap);
		v = Dot(ac, e) * ood;
		if (v < 0.0f || v > 1.0f) // ...here...
			return false;
		w = -Dot(ab, e) * ood;
		if (w < 0.0f || v + w > 1.0f) // ...and here
			return false;

		u = 1.0f - v - w;
		if (normal)
			*normal = n;
		sign = d;

		return true;
	}

	// mostly taken from Real Time Collision Detection - p192
	CUDA_CALLABLE inline bool IntersectRayTri(const ::Mirage::Vec3&p, const ::Mirage::Vec3&dir, const ::Mirage::Vec3&a, const ::Mirage::Vec3&b, const ::Mirage::Vec3&c, float &t, float &u, float &v, float &w, ::Mirage::Vec3*normal)
	{
		const ::Mirage::Vec3 ab = b - a;
		const ::Mirage::Vec3 ac = c - a;

		// calculate normal
		::Mirage::Vec3 n = Cross(ab, ac);

		// need to solve a system of three equations to give t, u, v
		float d = Dot(-dir, n);

		// if dir is parallel to triangle plane or points away from triangle
		if (d <= 0.0f)
			return false;

		::Mirage::Vec3 ap = p - a;
		t = Dot(ap, n);

		// ignores tris behind
		if (t < 0.0f)
			return false;

		// compute barycentric coordinates
		::Mirage::Vec3 e = Cross(-dir, ap);
		v = Dot(ac, e);
		if (v < 0.0f || v > d)
			return false;

		w = -Dot(ab, e);
		if (w < 0.0f || v + w > d)
			return false;

		float ood = 1.0f / d;
		t *= ood;
		v *= ood;
		w *= ood;
		u = 1.0f - v - w;

		// optionally write out normal (todo: this branch is a performance concern, should probably remove)
		if (normal)
			*normal = Normalize(n);

		return true;
	}

	CUDA_CALLABLE CUDA_CALLABLE inline float ScalarTriple(const ::Mirage::Vec3&a, const ::Mirage::Vec3&b, const ::Mirage::Vec3&c) { return Dot(Cross(a, b), c); }

	// mostly taken from Real Time Collision Detection - p192
	CUDA_CALLABLE CUDA_CALLABLE inline bool IntersectSegmentTri(const ::Mirage::Vec3&p, const ::Mirage::Vec3&q, const ::Mirage::Vec3&a, const ::Mirage::Vec3&b, const ::Mirage::Vec3&c, float &t, float &u, float &v, float &w, ::Mirage::Vec3*normal, float expand)
	{
		const ::Mirage::Vec3 pq = q - p;
		const ::Mirage::Vec3 pa = a - p;
		const ::Mirage::Vec3 pb = b - p;
		const ::Mirage::Vec3 pc = c - p;

		u = ScalarTriple(pq, pc, pb);
		if (u < 0.0f)
			return 0;

		v = ScalarTriple(pq, pa, pc);
		if (v < 0.0f)
			return 0;

		w = ScalarTriple(pq, pb, pa);
		if (w < 0.0f)
			return 0;

		return true;
	}

	// RTCD 5.1.5, page 142
	CUDA_CALLABLE inline ::Mirage::Vec3 Vec3ClosestPointOnTriangle(const ::Mirage::Vec3&a, const ::Mirage::Vec3&b, const ::Mirage::Vec3&c, const ::Mirage::Vec3 p)
	{
		::Mirage::Vec3 ab = b - a;
		::Mirage::Vec3 ac = c - a;
		::Mirage::Vec3 ap = p - a;

		float d1 = Dot(ab, ap);
		float d2 = Dot(ac, ap);
		if (d1 <= 0.0f && d2 <= 0.0f)
			return ::Mirage::Vec3(a);

		::Mirage::Vec3 bp = p - b;
		float d3 = Dot(ab, bp);
		float d4 = Dot(ac, bp);
		if (d3 >= 0.0f && d4 <= d3)
			return ::Mirage::Vec3(b);

		float vc = d1 * d4 - d3 * d2;
		if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
		{
			float v = d1 / (d1 - d3);
			return ::Mirage::Vec3(a + v * ab);
		}

		::Mirage::Vec3 cp = p - c;
		float d5 = Dot(ab, cp);
		float d6 = Dot(ac, cp);
		if (d5 * d6 >= 0.0f)
		{
			float v = d5 / (d5 + d6);
			return ::Mirage::Vec3(a + v * ab);
		}

		float vb = d5 * d2 - d1 * d6;
		if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
		{
			float w = d2 / (d2 - d6);
			return ::Mirage::Vec3(a + w * ac);
		}

		float va = d3 * d6 - d5 * d4;
		if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
		{
			float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			return ::Mirage::Vec3(b + w * (c - b));
		}

		float denom = 1.0f / (va + vb + vc);
		float v = vb * denom;
		float w = vc * denom;
		return ::Mirage::Vec3(a + ab * v + ac * w);
	}

	CUDA_CALLABLE CUDA_CALLABLE inline float SqDistPointSegment(::Mirage::Vec3 a, ::Mirage::Vec3 b, ::Mirage::Vec3 c)
	{
		::Mirage::Vec3 ab = b - a, ac = c - a, bc = c - b;
		float e = Dot(ac, ab);

		if (e <= 0.0f)
			return Dot(ac, ac);
		float f = Dot(ab, ab);

		if (e >= f)
			return Dot(bc, bc);

		return Dot(ac, ac) - e * e / f;
	}

	CUDA_CALLABLE CUDA_CALLABLE inline bool PointInTriangle(::Mirage::Vec3 a, ::Mirage::Vec3 b, ::Mirage::Vec3 c, ::Mirage::Vec3 p)
	{
		a -= p;
		b -= p;
		c -= p;

		::Mirage::Vec3 u = Cross(b, c);
		::Mirage::Vec3 v = Cross(c, a);

		if (Dot(u, v) <= 0.0f)
			return false;

		::Mirage::Vec3 w = Cross(a, b);

		if (Dot(u, w) <= 0.0f)
			return false;

		return true;
	}

	CUDA_CALLABLE inline float minf(const float a, const float b) { return a < b ? a : b; }
	CUDA_CALLABLE inline float maxf(const float a, const float b) { return a > b ? a : b; }

	// from Ompf
	CUDA_CALLABLE inline bool IntersectRayAABBFast(const ::Mirage::Vec3&pos, const ::Mirage::Vec3&rcp_dir, const ::Mirage::Vec3&min, const ::Mirage::Vec3&max, float &t)
	{

		float
			l1 = (min.x - pos.x) * rcp_dir.x,
			l2 = (max.x - pos.x) * rcp_dir.x,
			lmin = minf(l1, l2),
			lmax = maxf(l1, l2);

		l1 = (min.y - pos.y) * rcp_dir.y;
		l2 = (max.y - pos.y) * rcp_dir.y;
		lmin = maxf(minf(l1, l2), lmin);
		lmax = minf(maxf(l1, l2), lmax);

		l1 = (min.z - pos.z) * rcp_dir.z;
		l2 = (max.z - pos.z) * rcp_dir.z;
		lmin = maxf(minf(l1, l2), lmin);
		lmax = minf(maxf(l1, l2), lmax);

		// return ((lmax > 0.f) & (lmax >= lmin));
		// return ((lmax > 0.f) & (lmax > lmin));
		bool hit = ((lmax >= 0.f) & (lmax >= lmin));
		if (hit)
			t = lmin;
		return hit;
	}

	CUDA_CALLABLE inline bool IntersectRayAABB(const ::Mirage::Vec3&start, const ::Mirage::Vec3&dir, const ::Mirage::Vec3&min, const ::Mirage::Vec3&max, float &t, ::Mirage::Vec3*normal)
	{
		//! calculate candidate plane on each axis
		float tx = -1.0f, ty = -1.0f, tz = -1.0f;
		bool inside = true;

		//! use unrolled loops

		//! x
		if (start.x < min.x)
		{
			if (dir.x != 0.0f)
				tx = (min.x - start.x) / dir.x;
			inside = false;
		}
		else if (start.x > max.x)
		{
			if (dir.x != 0.0f)
				tx = (max.x - start.x) / dir.x;
			inside = false;
		}

		//! y
		if (start.y < min.y)
		{
			if (dir.y != 0.0f)
				ty = (min.y - start.y) / dir.y;
			inside = false;
		}
		else if (start.y > max.y)
		{
			if (dir.y != 0.0f)
				ty = (max.y - start.y) / dir.y;
			inside = false;
		}

		//! z
		if (start.z < min.z)
		{
			if (dir.z != 0.0f)
				tz = (min.z - start.z) / dir.z;
			inside = false;
		}
		else if (start.z > max.z)
		{
			if (dir.z != 0.0f)
				tz = (max.z - start.z) / dir.z;
			inside = false;
		}

		//! if point inside all planes
		if (inside)
		{
			t = 0.0f;
			return true;
		}

		//! we now have t values for each of possible intersection planes
		//! find the maximum to get the intersection point
		float tmax = tx;
		int taxis = 0;

		if (ty > tmax)
		{
			tmax = ty;
			taxis = 1;
		}
		if (tz > tmax)
		{
			tmax = tz;
			taxis = 2;
		}

		if (tmax < 0.0f)
			return false;

		//! check that the intersection point lies on the plane we picked
		//! we don't test the axis of closest intersection for precision reasons

		//! no eps for now
		float eps = 0.0f;

		::Mirage::Vec3 hit = start + dir * tmax;

		if ((hit.x < min.x - eps || hit.x > max.x + eps) && taxis != 0)
			return false;
		if ((hit.y < min.y - eps || hit.y > max.y + eps) && taxis != 1)
			return false;
		if ((hit.z < min.z - eps || hit.z > max.z + eps) && taxis != 2)
			return false;

		//! output results
		t = tmax;

		return true;
	}

	// construct a plane equation such that ax + by + cz + dw = 0
	CUDA_CALLABLE inline ::Mirage::Vec4 PlaneFromPoints(const ::Mirage::Vec3&p, const ::Mirage::Vec3&q, const ::Mirage::Vec3&r)
	{
		::Mirage::Vec3 e0 = q - p;
		::Mirage::Vec3 e1 = r - p;

		::Mirage::Vec3 n = SafeNormalize(Cross(e0, e1));

		return ::Mirage::Vec4(n.x, n.y, n.z, -Dot(p, n));
	}

	CUDA_CALLABLE inline bool IntersectPlaneAABB(const ::Mirage::Vec4 &plane, const ::Mirage::Vec3&center, const ::Mirage::Vec3&extents)
	{
		float radius = ::Mirage::Abs(extents.x * plane.x) + ::Mirage::Abs(extents.y * plane.y) + ::Mirage::Abs(extents.z * plane.z);
		float delta = Dot(center, Vec3(plane)) + plane.w;

		return ::Mirage::Abs(delta) <= radius;
	}

	//-----------------------
	// Templated query methods

	template <typename Func>
	CUDA_CALLABLE void QueryRayOld(const BVHNode *root, Func &f, const ::Mirage::Vec3&start, const ::Mirage::Vec3&dir)
	{
		::Mirage::Vec3 rcpDir;
		rcpDir.x = 1.0f / dir.x;
		rcpDir.y = 1.0f / dir.y;
		rcpDir.z = 1.0f / dir.z;

		int stack[32];
		stack[0] = 0;

		int count = 1;

		while (count)
		{
			const BVHNode *n = root + stack[--count];

#if CUDA
			// load using two 128 bit loads
			float4 f4[2];

			const float4 *__restrict__ fn = (const float4 *)(n);
			f4[0] = fn[0];
			f4[1] = fn[1];

			const BVHNode &node = (const BVHNode &)(f4[0]);
#else
			const BVHNode node = *n;
#endif

			float t;
			// if (IntersectRayAABB(start, dir, n->bounds.lower, n->bounds.upper, t, nullptr))
			if (IntersectRayAABBFast(start, rcpDir, node.bounds.lower, node.bounds.upper, t))
			{
				if (node.leaf)
				{
					f(node.leftIndex);
				}
				else
				{
					stack[count++] = node.leftIndex;
					stack[count++] = node.rightIndex;
				}
			}
		}
	}

	CUDA_CALLABLE inline BVHNode fetchNode(const BVHNode *ptr, int index)
	{
#if __CUDA_ARCH__ && USE_TEXTURES

		// load using two 128 bit loads
		float4 f4[2];

		f4[0] = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index * 2 + 0);
		f4[1] = tex1Dfetch<float4>((cudaTextureObject_t)ptr, index * 2 + 1);

		return (const BVHNode &)(f4[0]);
#else
		return ptr[index];
#endif
	}

	template <typename Func>
	CUDA_CALLABLE void QueryRay(const BVHNode *root, Func &f, const ::Mirage::Vec3&start, const ::Mirage::Vec3&dir)
	{
		::Mirage::Vec3 rcpDir;
		rcpDir.x = 1.0f / dir.x;
		rcpDir.y = 1.0f / dir.y;
		rcpDir.z = 1.0f / dir.z;

		int stack[32];
		stack[0] = 0;

		int count = 1;

		while (count)
		{
			BVHNode node = fetchNode(root, stack[--count]);

			if (node.leaf)
			{
				f(node.leftIndex);
			}
			else
			{
				// check children
				BVHNode left = fetchNode(root, node.leftIndex);
				BVHNode right = fetchNode(root, node.rightIndex);

				float tLeft = FLT_MAX;
				bool hitLeft = IntersectRayAABBFast(start, rcpDir, left.bounds.lower, left.bounds.upper, tLeft);

				float tRight = FLT_MAX;
				bool hitRight = IntersectRayAABBFast(start, rcpDir, right.bounds.lower, right.bounds.upper, tRight);

				if (hitLeft)
					stack[count++] = node.leftIndex;

				if (hitRight)
					stack[count++] = node.rightIndex;
			}
		}
	}

	struct MeshQuery
	{
		CUDA_CALLABLE inline MeshQuery(const MeshGeometry &m, const ::Mirage::Vec3&origin, const ::Mirage::Vec3&dir, float time) : mesh(m), rayOrigin(origin), rayDir(dir), rayTime(time), closestT(FLT_MAX) {}

		CUDA_CALLABLE inline void operator()(int i)
		{
			float t, u, v, w;
			::Mirage::Vec3 n;

			int i0 = ::Mirage::fetchInt(mesh.indices, i * 3 + 0);
			int i1 = ::Mirage::fetchInt(mesh.indices, i * 3 + 1);
			int i2 = ::Mirage::fetchInt(mesh.indices, i * 3 + 2);

			::Mirage::Vec3 a = fetchVec3(mesh.positions, i0);
			::Mirage::Vec3 b = fetchVec3(mesh.positions, i1);
			::Mirage::Vec3 c = fetchVec3(mesh.positions, i2);

			// Deforming (2-keyframe) motion blur: lerp this triangle's
			// vertices toward their end-of-shutter positions by rayTime,
			// mirroring the rigid Primitive::startTransform/endTransform
			// interpolation one level down, per-triangle instead of
			// per-object. mesh.positionsEnd is only non-null when
			// Mesh::verticesEnd was populated (see GeometryFromMesh) - a
			// static mesh takes the exact same path as before this feature.
			if (mesh.positionsEnd)
			{
				a = ::Mirage::Lerp(a, fetchVec3(mesh.positionsEnd, i0), rayTime);
				b = ::Mirage::Lerp(b, fetchVec3(mesh.positionsEnd, i1), rayTime);
				c = ::Mirage::Lerp(c, fetchVec3(mesh.positionsEnd, i2), rayTime);
			}

			float sign;
			// if (IntersectRayTri(rayOrigin, rayDir, a, b, c, t, u, v, w, &n))
			if (IntersectRayTriTwoSided(rayOrigin, rayDir, a, b, c, t, u, v, w, sign, &n))
			{
				if (t > 0.0f && t < closestT)
				{
					closestT = t;
					closestU = u;
					closestV = v;
					closestW = w;

					closestTri = i;
					closestNormal = n * sign;
				}
			}
		}

		const MeshGeometry &mesh;
		const ::Mirage::Vec3 rayOrigin;
		const ::Mirage::Vec3 rayDir;
		const float rayTime;

		float closestT;
		float closestU;
		float closestV;
		float closestW;

		::Mirage::Vec3 closestNormal;
		int closestTri;
	};

	CUDA_CALLABLE bool inline IntersectRayMesh(const MeshGeometry &mesh, const ::Mirage::Vec3&origin, const ::Mirage::Vec3&dir, float tmax, float rayTime, float &t, float &u, float &v, float &w, int &tri, ::Mirage::Vec3&triNormal)
	{

		MeshQuery query(mesh, origin, dir, rayTime);

		::Mirage::Vec3 rcpDir;
		rcpDir.x = 1.0f / dir.x;
		rcpDir.y = 1.0f / dir.y;
		rcpDir.z = 1.0f / dir.z;

		int stack[32];
		stack[0] = 0;

		int count = 1;

		while (count)
		{
			BVHNode node = fetchNode(mesh.nodes, stack[--count]);

			if (node.leaf)
			{
				query(node.leftIndex);

				// truncate ray
				tmax = query.closestT;
			}
			else
			{
				// check children
				BVHNode left = fetchNode(mesh.nodes, node.leftIndex);
				BVHNode right = fetchNode(mesh.nodes, node.rightIndex);

				float tLeft;
				bool hitLeft = IntersectRayAABBFast(origin, rcpDir, left.bounds.lower, left.bounds.upper, tLeft) && tLeft < tmax;

				float tRight;
				bool hitRight = IntersectRayAABBFast(origin, rcpDir, right.bounds.lower, right.bounds.upper, tRight) && tRight < tmax;

				// traverse closest first
				if (hitLeft && hitRight && (tLeft < tRight))
				{
					int tmp = node.leftIndex;
					node.leftIndex = node.rightIndex;
					node.rightIndex = tmp;
				}

				if (hitLeft)
					stack[count++] = node.leftIndex;

				if (hitRight)
					stack[count++] = node.rightIndex;
			}
		}

		// output results
		if (query.closestT < FLT_MAX)
		{
			t = query.closestT;
			u = query.closestU;
			v = query.closestV;
			w = query.closestW;
			tri = query.closestTri;
			triNormal = query.closestNormal;

			return true;
		}
		else
		{
			return false;
		}
	}

	template <typename T>
	CUDA_CALLABLE inline void QueryBVH(T &callback, BVHNode *root, const ::Mirage::Vec3&origin, const ::Mirage::Vec3&dir)
	{
		::Mirage::Vec3 rcpDir;
		rcpDir.x = 1.0f / dir.x;
		rcpDir.y = 1.0f / dir.y;
		rcpDir.z = 1.0f / dir.z;

		int stack[32];
		stack[0] = 0;

		int count = 1;

		while (count)
		{
			BVHNode node = fetchNode(root, stack[--count]);

			if (node.leaf)
			{
				callback(node.leftIndex);
			}
			else
			{
				// check children
				BVHNode left = fetchNode(root, node.leftIndex);
				BVHNode right = fetchNode(root, node.rightIndex);

				float tLeft;
				bool hitLeft = IntersectRayAABBFast(origin, rcpDir, left.bounds.lower, left.bounds.upper, tLeft);

				float tRight;
				bool hitRight = IntersectRayAABBFast(origin, rcpDir, right.bounds.lower, right.bounds.upper, tRight);

				// traverse closest first
				if (hitLeft && hitRight && (tLeft < tRight))
				{
					int tmp = node.leftIndex;
					node.leftIndex = node.rightIndex;
					node.rightIndex = tmp;
				}

				if (hitLeft)
					stack[count++] = node.leftIndex;

				if (hitRight)
					stack[count++] = node.rightIndex;
			}
		}
	}

	CUDA_CALLABLE bool inline IntersectRayMeshOld(const MeshGeometry &mesh, const ::Mirage::Vec3&origin, const ::Mirage::Vec3&dir, float tmax, float &t, float &u, float &v, float &w, int &tri, ::Mirage::Vec3&triNormal)
	{

		// Legacy/unused (see QueryRayOld precedent above) - not motion-blur
		// aware, always samples the start-of-shutter positions.
		MeshQuery query(mesh, origin, dir, 0.0f);

		QueryRay(mesh.nodes, query, origin, dir);

		if (query.closestT < FLT_MAX)
		{
			t = query.closestT;
			u = query.closestU;
			v = query.closestV;
			w = query.closestW;
			tri = query.closestTri;
			triNormal = query.closestNormal;

			return true;
		}
		else
		{
			return false;
		}
	}

	//-------------
	// pdf that a point and dir were sampled by the light (assuming the ray hits the shape)

	CUDA_CALLABLE inline float PrimitiveArea(const Primitive &p)
	{
		switch (p.type)
		{
		case eSphere:
		{
			float area = 4.0f * kPi * p.sphere.radius * p.sphere.radius;
			return area;
		}
		case ePlane:
		{
			return 0.0f;
		}
		case eMesh:
		{
			return p.mesh.area * p.endTransform.s;
		}
		};

		return 0.0f;
	}

	CUDA_CALLABLE inline void PrimitiveSample(const Primitive &p, float time, ::Mirage::Vec3&pos, ::Mirage::Vec3&normal, ::Mirage::Random &rand)
	{
		Transform transform = InterpolateTransform(p.startTransform, p.endTransform, time);

		switch (p.type)
		{
		case eSphere:
		{
			float u1, u2;
			Sample2D(rand, u1, u2);

			// todo: handle scaling in transform matrix
			pos = TransformPoint(transform, UniformSampleSphere(u1, u2) * p.sphere.radius);
			normal = Normalize(pos - transform.p);
			return;
		}
		case ePlane:
		{
			assert(0);
			return;
		}
		case eMesh:
		{
			// Validate mesh data before accessing
			if (!p.mesh.positions || !p.mesh.indices || !p.mesh.normals || !p.mesh.cdf) {
				// Return default values if mesh data is invalid
				pos = ::Mirage::Vec3(0.0f);
				normal = ::Mirage::Vec3(0.0f, 1.0f, 0.0f);
				return;
			}

			float r = rand.Randf();

			const float *triPtr = LowerBound(p.mesh.cdf, p.mesh.cdf + p.mesh.numIndices / 3, r);
			int tri = ::Mirage::Min(int(triPtr - p.mesh.cdf), p.mesh.numIndices / 3 - 1);

			const ::Mirage::Vec2 uv = fetchVec2(p.mesh.uvs, tri * 3);
			float u = uv[0];
			float v = uv[1];
			// UniformSampleTriangle(rand, u, v);

			// interpolate tri data
			int i0 = ::Mirage::fetchInt(p.mesh.indices, tri * 3 + 0);
			int i1 = ::Mirage::fetchInt(p.mesh.indices, tri * 3 + 1);
			int i2 = ::Mirage::fetchInt(p.mesh.indices, tri * 3 + 2);

			const ::Mirage::Vec3 a = fetchVec3(p.mesh.positions, i0);
			const ::Mirage::Vec3 b = fetchVec3(p.mesh.positions, i1);
			const ::Mirage::Vec3 c = fetchVec3(p.mesh.positions, i2);

			const ::Mirage::Vec3 n1 = fetchVec3(p.mesh.normals, i0);
			const ::Mirage::Vec3 n2 = fetchVec3(p.mesh.normals, i1);
			const ::Mirage::Vec3 n3 = fetchVec3(p.mesh.normals, i2);

			pos = TransformPoint(transform, u * a + v * b + (1.0f - u - v) * c);
			normal = SafeNormalize(TransformVector(transform, u * n1 + v * n2 + (1.0f - u - v) * n3));
			return;
		}
		}
	}

	CUDA_CALLABLE inline Bounds PrimitiveBounds(const Primitive &p)
	{
		Bounds localBounds;

		switch (p.type)
		{
		case eSphere:
		{
			localBounds.lower = -p.sphere.radius;
			localBounds.upper = p.sphere.radius;
			break;
		}
		case ePlane:
		{
			localBounds.lower = -1.e+8f;
			localBounds.upper = 1.e+8f;
			break;
		}
		case eMesh:
		{
			// read bounds from bvh root
			if (p.mesh.nodes && p.mesh.numNodes > 0) {
				localBounds = p.mesh.nodes[0].bounds;
			} else {
				// No BVH, calculate bounds from vertices directly
				if (p.mesh.positions && p.mesh.numVertices > 0) {
					localBounds.lower = p.mesh.positions[0];
					localBounds.upper = p.mesh.positions[0];
					for (int i = 1; i < p.mesh.numVertices; ++i) {
						localBounds.AddPoint(p.mesh.positions[i]);
					}
				} else {
					// No geometry data, return empty bounds
					localBounds.lower = Vec3(0.0f);
					localBounds.upper = Vec3(0.0f);
				}
			}
			break;
		}
		};

		// transform bounds based on start/end
		Bounds startBounds = TransformBounds(p.startTransform, localBounds);
		Bounds endBounds = TransformBounds(p.endTransform, localBounds);

		return Union(startBounds, endBounds);
	}

	struct Ray
	{
		CUDA_CALLABLE inline Ray(const ::Mirage::Vec3&o, const ::Mirage::Vec3&d, float t) : origin(o), dir(d), time(t) {}

		::Mirage::Vec3 origin;
		::Mirage::Vec3 dir;

		float time;
	};

	// outUV is only populated for eMesh hits on a mesh that has UV data
	// (MeshGeometry::uvs != nullptr, i.e. Mesh::uvs was populated by whatever
	// built the scene) - left untouched otherwise, so callers should treat an
	// unmodified default (typically Vec2(0,0), the caller's initial value) as
	// "no UV available for this hit", not a real coordinate. Sphere/plane hits
	// never populate outUV (see the plan's Deferrals - no procedural UV for
	// non-mesh primitives in this milestone). outTangent follows the exact
	// same convention, gated on MeshGeometry::tangents != nullptr
	// (Mesh::HasTangents()) instead of uvs.
	CUDA_CALLABLE inline bool PrimitiveIntersect(const Primitive &p, const Ray &ray, float &outT, ::Mirage::Vec3*outNormal, ::Mirage::Vec2 *outUV = nullptr, ::Mirage::Vec3 *outTangent = nullptr)
	{
		Transform transform = InterpolateTransform(p.startTransform, p.endTransform, ray.time);

		switch (p.type)
		{
		case eSphere:
		{
			float minT, maxT;
			::Mirage::Vec3 n;

			bool hit = IntersectRaySphere(transform.p, p.sphere.radius * transform.s, ray.origin, ray.dir, minT, maxT, &n);

			if (hit)
			{
				outT = minT;
				*outNormal = n;
			}
			return hit;
		}
		case ePlane:
		{
			bool hit = IntersectRayPlane(ray.origin, ray.dir, (const ::Mirage::Vec4 &)p.plane, outT);

			if (hit && outNormal)
			{
				*outNormal = (const ::Mirage::Vec3&)p.plane;
			}

			return hit;
		}
		case eMesh:
		{
			// Validate mesh data before accessing
			if (!p.mesh.positions || !p.mesh.indices || !p.mesh.normals || !p.mesh.nodes) {
				return false; // No valid mesh data to intersect with
			}

			::Mirage::Vec3 localOrigin = InverseTransformPoint(transform, ray.origin);
			::Mirage::Vec3 localDir = InverseTransformVector(transform, ray.dir);

			float t, u, v, w;
			int tri;
			::Mirage::Vec3 triNormal;

			// transform ray to mesh space
			bool hit = IntersectRayMesh(p.mesh, localOrigin, localDir, FLT_MAX, ray.time, t, u, v, w, tri, triNormal);

			if (hit)
			{
				// interpolate vertex normals
				int i0 = ::Mirage::fetchInt(p.mesh.indices, tri * 3 + 0);
				int i1 = ::Mirage::fetchInt(p.mesh.indices, tri * 3 + 1);
				int i2 = ::Mirage::fetchInt(p.mesh.indices, tri * 3 + 2);

				const ::Mirage::Vec3 n1 = fetchVec3(p.mesh.normals, i0);
				const ::Mirage::Vec3 n2 = fetchVec3(p.mesh.normals, i1);
				const ::Mirage::Vec3 n3 = fetchVec3(p.mesh.normals, i2);

				::Mirage::Vec3 smoothNormal = u * n1 + v * n2 + w * n3;

				// ensure smooth normal lies on the same side of the geometric normal
				if (Dot(smoothNormal, triNormal) < 0.0f)
					smoothNormal *= -1.0f;

				outT = t;
				*outNormal = SafeNormalize(TransformVector(transform, smoothNormal), triNormal);

				// same u/v/w barycentric weights (matched to i0/i1/i2) used for
				// the smooth-normal interpolation above, applied to per-vertex
				// UVs instead - see the plan's "UV threading" section.
				if (outUV && p.mesh.uvs)
				{
					const ::Mirage::Vec2 uv0 = fetchVec2(p.mesh.uvs, i0);
					const ::Mirage::Vec2 uv1 = fetchVec2(p.mesh.uvs, i1);
					const ::Mirage::Vec2 uv2 = fetchVec2(p.mesh.uvs, i2);
					*outUV = u * uv0 + v * uv1 + w * uv2;
				}

				// Same u/v/w barycentric weights again, applied to
				// per-vertex tangents - mirrors the outUV block immediately
				// above exactly, gated on tangent data instead of UV data.
				if (outTangent && p.mesh.tangents)
				{
					const ::Mirage::Vec3 t1 = fetchVec3(p.mesh.tangents, i0);
					const ::Mirage::Vec3 t2 = fetchVec3(p.mesh.tangents, i1);
					const ::Mirage::Vec3 t3 = fetchVec3(p.mesh.tangents, i2);
					::Mirage::Vec3 smoothTangent = u * t1 + v * t2 + w * t3;
					*outTangent = SafeNormalize(TransformVector(transform, smoothTangent));
				}
			}

			return hit;
		}
		}

		return false;
	}
}