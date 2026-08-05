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
            geo.positionsEnd = nullptr;
            geo.tangents = nullptr;
            geo.numNodes = 0;
            geo.numIndices = 0;
            geo.numVertices = 0;
            geo.area = 0.0f;
            geo.id = 0;
            geo.meshIndex = -1;
            return geo;
        }

        geo.positions = &mesh->vertices[0];
        geo.normals = mesh->normals.empty() ? nullptr : &mesh->normals[0];
        geo.indices = mesh->indices.empty() ? nullptr : &mesh->indices[0];
        geo.uvs = mesh->uvs.empty() ? nullptr : &mesh->uvs[0];
        geo.nodes = (mesh->bvh.numNodes > 0 && mesh->bvh.nodes) ? &mesh->bvh.nodes[0] : nullptr;
        geo.cdf = mesh->cdf.empty() ? nullptr : &mesh->cdf[0];
        geo.positionsEnd = mesh->HasMotion() ? &mesh->verticesEnd[0] : nullptr;
        geo.tangents = mesh->HasTangents() ? &mesh->tangents[0] : nullptr;
        geo.area = mesh->area;

        geo.numNodes = mesh->bvh.numNodes;
        geo.numIndices = mesh->indices.size();
        geo.numVertices = mesh->vertices.size();

        geo.id = (unsigned long)mesh;
        geo.meshIndex = -1; // resolved by the scene owner - see MeshGeometry::meshIndex

        return geo;
    }

    // filmic: the Hejl/Burgess-Dawson filmic curve (these exact 6.2/0.5/1.7/
    // 0.06 constants, with the 0.004 toe offset) - a well-known property of
    // *this specific curve* is that gamma correction is already baked into
    // its polynomial fit, so its output is display-ready/sRGB-encoded
    // directly with no separate encode step
    // (https://filmicworlds.com/blog/filmic-tonemapping-operators/, the
    // "you don't need to convert to sRGB after" derivation). The previous
    // code called SrgbToLinear - a *decode* - on this already-encoded
    // output, double-transforming (darkening/distorting) every LDR render;
    // removed rather than replaced, since no gamma call belongs here at all.
    //
    // `limit` is now applied as a pre-curve highlight-compression control
    // (the curve's "linear white point": scaling input down before the
    // curve pushes highlight roll-off further out) - previously a dead
    // parameter, unused in the function body.
    CUDA_CALLABLE inline Color ToneMap(const Color &c, float limit)
    {
        Vec3 texColor = Vec3(c) / Max(limit, 1e-4f);
        Vec3 x = Max(Vec3(0.0f), texColor - Vec3(0.004f));
        Vec3 retColor = (x * (Vec3(6.2f) * x + Vec3(.5f))) / (x * (Vec3(6.2f) * x + Vec3(1.7f)) + Vec3(0.06));

        return Color(retColor, c.w);
    }

    // Krzysztof Narkowicz's fitted ACES RRT+ODT approximation
    // (https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/).
    // Unlike ToneMap above, this curve's output is still display-*linear* -
    // Narkowicz's own writeup calls for a standard gamma/sRGB encode after
    // it, which is what LinearToSrgb (an *encode*) provides here; the
    // previous code called SrgbToLinear (a *decode*), the wrong direction.
    // Same `limit` pre-curve highlight-compression treatment as ToneMap.
    CUDA_CALLABLE inline Color ToneMapACES(const Color &c, float limit)
    {
        Vec3 texColor = Vec3(c) / Max(limit, 1e-4f);
        Vec3 x = Max(Vec3(0.0f), texColor - Vec3(0.004f));
        Vec3 retColor = (x * (Vec3(2.51f) * x + Vec3(0.03f))) / (x * (Vec3(2.43f) * x + Vec3(0.59f)) + Vec3(0.14f));

        return LinearToSrgb(Color(retColor, c.w));
    }

    // Minimal color-management extension point (docs/PRODUCTION_READINESS.md's
    // Tier 3 "no color management" finding) - deliberately not a full OCIO
    // integration (config parsing, ACES IDT/RRT/ODT reference chain, look/LUT
    // management, display calibration are all explicitly out of scope here;
    // OCIO's own dependency chain, e.g. yaml-cpp, is disproportionate for a
    // codebase that avoids even linking real OpenEXR in favor of vendored
    // tinyexr). This enum + ApplyViewTransform is the single place a real
    // OCIO backend could later plug in without touching every call site.
    enum class ViewTransform
    {
        eNone,        // raw linear values, exposure-scaled only - no display
                      // encode at all; only meaningful for float/EXR output,
                      // never for an 8-bit target (would look far too dark).
        eFilmic,      // ToneMap: Hejl/Burgess-Dawson filmic curve.
        eAcesLike,    // ToneMapACES: Narkowicz's fitted ACES approximation.
        eSrgbDisplay, // plain LinearToSrgb encode, no filmic highlight roll-off.
    };

    // Maps a linear-light pixel to a display-referred one for a given
    // ViewTransform, applying `exposure` as a multiplicative scale *before*
    // the transform (standard convention - matches Options::exposure /
    // Camera::ComputeExposureMultiplier's existing multiplicative
    // semantics elsewhere in the codebase). ToneMap/ToneMapACES's own
    // `limit` (highlight-compression) parameter is fixed at 1.0 here - no
    // additional knob beyond `exposure` is exposed at this layer today.
    CUDA_CALLABLE inline Color ApplyViewTransform(const Color &linear, ViewTransform vt, float exposure)
    {
        Color exposed = Color(Vec3(linear) * exposure, linear.w);
        switch (vt)
        {
        case ViewTransform::eFilmic:
            return ToneMap(exposed, 1.0f);
        case ViewTransform::eAcesLike:
            return ToneMapACES(exposed, 1.0f);
        case ViewTransform::eSrgbDisplay:
            return LinearToSrgb(exposed);
        case ViewTransform::eNone:
        default:
            return exposed;
        }
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
