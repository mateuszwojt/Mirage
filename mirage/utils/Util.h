#pragma once

#include "mirage/prims/Mesh.h"
#include "mirage/core/Scene.h"

namespace Mirage
{
    inline MeshGeometry GeometryFromMesh(const Mesh *mesh)
    {
        MeshGeometry geo;
        
        // Validate mesh data before accessing
        if (!mesh || mesh->vertices.empty()) {
            // Return empty geometry for invalid mesh
            geo.positions = nullptr;
            geo.normals = nullptr;
            geo.indices = nullptr;
            geo.uvs = nullptr;
            geo.nodes = nullptr;
            geo.cdf = nullptr;
            geo.numNodes = 0;
            geo.numIndices = 0;
            geo.numVertices = 0;
            geo.area = 0.0f;
            geo.id = 0;
            return geo;
        }
        
        geo.positions = &mesh->vertices[0];
        geo.normals = mesh->normals.empty() ? nullptr : &mesh->normals[0];
        geo.indices = mesh->indices.empty() ? nullptr : &mesh->indices[0];
        geo.uvs = mesh->uvs.empty() ? nullptr : &mesh->uvs[0];
        geo.nodes = (mesh->bvh.numNodes > 0 && mesh->bvh.nodes) ? &mesh->bvh.nodes[0] : nullptr;
        geo.cdf = mesh->cdf.empty() ? nullptr : &mesh->cdf[0];
        geo.area = mesh->area;

        geo.numNodes = mesh->bvh.numNodes;
        geo.numIndices = mesh->indices.size();
        geo.numVertices = mesh->vertices.size();

        geo.id = (unsigned long)mesh;

        return geo;
    }

    CUDA_CALLABLE inline Color ToneMap(const Color &c, float limit)
    {
        // filmic
        Vec3 texColor = Vec3(c);
        Vec3 x = Max(Vec3(0.0f), texColor - Vec3(0.004f));
        Vec3 retColor = (x * (Vec3(6.2f) * x + Vec3(.5f))) / (x * (Vec3(6.2f) * x + Vec3(1.7f)) + Vec3(0.06));

        return SrgbToLinear(Color(retColor, 0.0f));
    }

    CUDA_CALLABLE inline Color ToneMapACES(const Color &c, float limit)
    {
        Vec3 texColor = Vec3(c);
        Vec3 x = Max(Vec3(0.0f), texColor - Vec3(0.004f));
        Vec3 retColor = (x * (Vec3(2.51f) * x + Vec3(0.03f))) / (x * (Vec3(2.43f) * x + Vec3(0.59f)) + Vec3(0.14f));

        return SrgbToLinear(Color(retColor, 0.0f));
    }

    struct CameraSampler
    {
        CUDA_CALLABLE inline CameraSampler() {}

        CUDA_CALLABLE inline CameraSampler(
            const Mat44 &cameraToWorld,
            float fov,
            float near,
            float far,
            float aperture,
            float focalPoint,
            bool enableDOF,
            int width,
            int height) : cameraToWorld(cameraToWorld), aperture(aperture), focalPoint(focalPoint), enableDOF(enableDOF)
        {
            Mat44 rasterToScreen(2.0f / width, 0.0f, 0.0f, -1.0f,
                                 0.0f, -2.0f / height, 0.0f, 1.0f,
                                 0.0f, 0.0f, 1.0f, 1.0f,
                                 0.0f, 0.0f, 0.0f, 1.0f);

            float f = tanf(fov * 0.5f);
            float aspect = float(width) / height;

            Mat44 screenToCamera(f * aspect, 0.0f, 0.0f, 0.0f,
                                 0.0f, f, 0.0f, 0.0f,
                                 0.0f, 0.0f, -1.0f, 0.0f,
                                 0.0f, 0.0f, 0.0f, 1.0f);

            rasterToWorld = cameraToWorld * screenToCamera * rasterToScreen;
        }

        CUDA_CALLABLE inline void GenerateRay(float rasterX, float rasterY, Vec3 &origin, Vec3 &dir)
        {
            Vec3 p = TransformPoint(rasterToWorld, Vec3(rasterX, rasterY, 0.0f));

            origin = Vec3(cameraToWorld.GetCol(3));
            dir = Normalize(p - origin);

            // Thin Lens Approximation
            if (enableDOF)
            {
                // get uniformly distributed points on the unit disk
                Vec2 lens;
                ConcentricDiskSample(xor128() / 4294967296.0f, xor128() / 4294967296.0f, &lens);

                // scale points in [-1, 1] domain to actual aperture radius
                lens *= (aperture / 2);

                Vec3 w = origin - dir;
                Vec3 u = Cross(Vec3(0, 1, 0), w);
                Vec3 v = Cross(w, u);
                Vec3 offset = u * lens.x + v * lens.y;

                // new origin is these points on the lens
                origin += offset;

                Vec3 focusPoint = focalPoint * dir;
                dir = Normalize(focusPoint - offset);
            }
        }

        CUDA_CALLABLE inline void ConcentricDiskSample(float ox, float oy, Vec2 *lens)
        {
            float phi, r;

            // switch coordinate space from [0, 1] to [-1, 1]
            float a = 2.0 * ox - 1.0;
            float b = 2.0 * oy - 1.0;

            if ((a * a) > (b * b))
            {
                r = a;
                phi = (0.78539816339f) * (b / a);
            }
            else
            {
                r = b;
                phi = (kPiOver2) - (0.78539816339f) * (a / b);
            }

            lens->x = r * cosf(phi);
            lens->y = r * sinf(phi);
        }

        // xorshift fast random number generator
        uint32_t xor128(void)
        {
            static uint32_t x = 123456789, y = 362436069, z = 521288629, w = 88675123;
            uint32_t t = x ^ (x << 11);
            x = y;
            y = z;
            z = w;
            return w = (w ^ (w >> 19) ^ t ^ (t >> 8));
        }

        Mat44 cameraToWorld;
        Mat44 rasterToWorld;
        float aperture;
        float focalPoint;
        float fov;
        bool enableDOF;
    };

    inline void MakeRelativePath(const char *filePath, const char *fileRelativePath, char *fullPath)
    {
        // get base path of file
        const char *lastSlash = nullptr;

        if (!lastSlash)
            lastSlash = strrchr(filePath, '\\');
        if (!lastSlash)
            lastSlash = strrchr(filePath, '/');

        int baseLength = 0;

        if (lastSlash)
        {
            baseLength = (lastSlash - filePath) + 1;

            // copy base path (including slash to relative path)
            memcpy(fullPath, filePath, baseLength);
        }

        // append mesh filename
        strcpy(fullPath + baseLength, fileRelativePath);
    }
}
