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

		std::string meshName;
		float area;

		void AddMesh(Mesh &m);

		void DuplicateVertex(size_t i);

		void CalculateNormals();
		void Transform(const Mat44 &m);
		void Normalize(float s = 1.0f); // scale so bounds in any dimension equals s and lower bound = (0,0,0)

		void GetBounds(Vec3 &minExtents, Vec3 &maxExtents) const;

		void rebuildBVH();
		void rebuildCDF();

		BVH bvh;
	};
}
