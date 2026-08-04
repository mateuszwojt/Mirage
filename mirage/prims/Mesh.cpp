#include "mirage/prims/Mesh.h"

#include <map>
#include <fstream>
#include <iostream>
#include <string>

namespace Mirage
{
    void Mesh::DuplicateVertex(size_t i)
    {
        assert(vertices.size() > i);
        vertices.push_back(vertices[i]);

        if (normals.size() > i)
            normals.push_back(normals[i]);
    }

    void Mesh::Normalize(float s)
    {
        Vec3 lower, upper;
        GetBounds(lower, upper);
        Vec3 edges = upper - lower;

        Transform(TranslationMatrix(Vec3(-lower)));

        float maxEdge = std::max(edges.x, std::max(edges.y, edges.z));
        Transform(ScaleMatrix(s / maxEdge));
    }

    void Mesh::CalculateNormals()
    {
        normals.resize(0);
        normals.resize(vertices.size());

        int numTris = indices.size() / 3;

        for (int i = 0; i < numTris; ++i)
        {
            int a = indices[i * 3 + 0];
            int b = indices[i * 3 + 1];
            int c = indices[i * 3 + 2];

            Vec3 n = Cross(vertices[b] - vertices[a], vertices[c] - vertices[a]);

            normals[a] += n;
            normals[b] += n;
            normals[c] += n;
        }

        int numVertices = int(vertices.size());

        for (int i = 0; i < numVertices; ++i)
            normals[i] = SafeNormalize(normals[i]);
    }

    void Mesh::rebuildBVH()
    {
        const int numTris = indices.size() / 3;

        if (numTris)
        {
            std::vector<Bounds> triangleBounds(numTris);

            for (int i = 0; i < numTris; ++i)
            {
                const Vec3 a = vertices[indices[i * 3 + 0]];
                const Vec3 b = vertices[indices[i * 3 + 1]];
                const Vec3 c = vertices[indices[i * 3 + 2]];

                triangleBounds[i].AddPoint(a);
                triangleBounds[i].AddPoint(b);
                triangleBounds[i].AddPoint(c);
            }

            BVHBuilder builder;
            bvh = builder.Build(&triangleBounds[0], numTris);
        }

        rebuildCDF();
    }

    void Mesh::rebuildCDF()
    {
        int numTris = indices.size() / 3;

        float totalArea = 0.0f;
        cdf.resize(0);

        for (int i = 0; i < numTris; ++i)
        {
            const Vec3 a = vertices[indices[i * 3 + 0]];
            const Vec3 b = vertices[indices[i * 3 + 1]];
            const Vec3 c = vertices[indices[i * 3 + 2]];

            const float area = 0.5f * Length(Cross(b - a, c - a));

            totalArea += area;

            cdf.push_back(totalArea);
        }

        // normalize cdf
        for (int i = 0; i < numTris; ++i)
        {
            cdf[i] /= totalArea;
        }

        // save total area
        area = totalArea;
    }

    void Mesh::AddMesh(Mesh &m)
    {
        int offset = vertices.size();

        // add new vertices
        vertices.insert(vertices.end(), m.vertices.begin(), m.vertices.end());
        normals.insert(normals.end(), m.normals.begin(), m.normals.end());

        // add new indices with offset
        for (size_t i = 0; i < m.indices.size(); ++i)
        {
            indices.push_back(m.indices[i] + offset);
        }
    }

    void Mesh::Transform(const Mat44 &m)
    {
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            vertices[i] = TransformPoint(m, vertices[i]);
            normals[i] = TransformVector(m, normals[i]);
        }
    }

    void Mesh::GetBounds(Vec3 &outMinExtents, Vec3 &outMaxExtents) const
    {
        Vec3 minExtents(REAL_MAX);
        Vec3 maxExtents(-REAL_MAX);

        // calculate face bounds
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            const Vec3 &a = vertices[i];

            minExtents = Min(a, minExtents);
            maxExtents = Max(a, maxExtents);
        }

        outMinExtents = Vec3(minExtents);
        outMaxExtents = Vec3(maxExtents);
    }
}