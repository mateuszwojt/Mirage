#pragma once

#include "mirage/math/Mat44.h"
#include "mirage/core/BVH.h"

#include <vector>

namespace Mirage
{
	class Mesh
	{
	public:
		Mesh() {};
		~Mesh() { delete[] bvh.nodes; }

		// Mesh Data
		std::vector<Vec3> vertices;
		std::vector<Vec3> normals;
		std::vector<Vec2> uvs;
		std::vector<int> indices;
		std::vector<float> cdf;

		// Deforming (2-keyframe) motion blur: end-of-shutter vertex snapshot,
		// parallel to vertices above (same index space). Empty (the default)
		// means "static mesh, no deformation" - every existing mesh-authoring
		// path that never touches this field keeps rendering exactly as
		// before this feature. AddMesh()/DuplicateVertex() do not merge or
		// extend it yet - only meshes authored with both keyframes up front
		// are supported. Shading normals are NOT deformed (still
		// barycentric-interpolated from the static `normals` array at hit
		// time, regardless of rayTime) - a v1 scope limit, not an oversight:
		// a moving/deforming triangle's correct shading normal would need
		// its own end-of-shutter snapshot and lerp, deferred to a follow-up
		// since it only affects shading quality, not hit position/silhouette
		// motion (what this feature targets).
		std::vector<Vec3> verticesEnd;

		// Tangent-space normal mapping: per-vertex tangent, parallel to
		// vertices/normals above (same index space). Empty (the default)
		// means "no tangent-space basis" - every existing mesh-authoring
		// path that never calls ComputeTangents() keeps rendering exactly
		// as before this feature (normalTextureIndex is simply never
		// sampled for such a mesh - see Renderer.cpp's shadedMaterial hook).
		std::vector<Vec3> tangents;

		std::string meshName;
		float area;

		// True only when verticesEnd was populated with one end-of-shutter
		// position per start-of-shutter vertex - a size mismatch (e.g. only
		// partially populated) is treated as "no motion" rather than an
		// out-of-bounds read.
		bool HasMotion() const { return !verticesEnd.empty() && verticesEnd.size() == vertices.size(); }

		// True only when ComputeTangents() successfully populated one
		// tangent per vertex - same "malformed data degrades safely"
		// convention as HasMotion().
		bool HasTangents() const { return !tangents.empty() && tangents.size() == vertices.size(); }

		void AddMesh(Mesh &m);

		void DuplicateVertex(size_t i);

		void CalculateNormals();

		// Per-vertex tangent-space basis for normal mapping, derived from
		// per-triangle UV derivatives (the standard Lengyel/"MikkTSpace-ish"
		// construction) then Gram-Schmidt-orthonormalized against the
		// already-computed vertex normal - requires CalculateNormals() to
		// have run first (normals.size() must already match vertices.size();
		// this bails out, leaving tangents empty/HasTangents()==false,
		// otherwise). No-ops (leaves tangents empty) if the mesh has no UV
		// data - tangent space is undefined without a UV parameterization.
		void ComputeTangents();

		void Transform(const Mat44 &m);
		void Normalize(float s = 1.0f); // scale so bounds in any dimension equals s and lower bound = (0,0,0)

		void GetBounds(Vec3 &minExtents, Vec3 &maxExtents) const;

		void rebuildBVH();
		void rebuildCDF();

		BVH bvh;
	};
}
