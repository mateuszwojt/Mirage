#pragma once

// Tangent-space normal mapping. Deliberately its own header, not folded into
// TextureSampling.h - this isn't a texture-sampling function (that part is
// just SampleTextureRGB, reused as-is, since a normal map's raw [0,1] RGB
// texel needs no special sampling logic of its own), it's the decode + TBN
// transform + face-forward safety net shared by both backends' shading hook
// (CPU: Renderer.cpp's PathTrace() shadedMaterial block; GPU: the mirrored
// NormalMapping.slang + PathTrace.slang hook).

#include "mirage/math/Vec3.h"
#include "mirage/utils/API.h"

namespace Mirage
{
    // `normal`/`tangent` are the already-interpolated, already-world-space
    // smooth shading normal and vertex tangent at the hit (see
    // Intersection.h's PrimitiveIntersect outTangent). `texel` is the raw
    // [0,1] RGB sample from the bound normal-map texture - callers must load
    // that texture through TextureColorSpace::eLinear (see TextureLoader.h);
    // a normal map is not a color and must never go through the sRGB decode
    // the albedo slot uses.
    CUDA_CALLABLE inline Vec3 ApplyNormalMap(const Vec3 &normal, const Vec3 &tangent, const Vec3 &texel)
    {
        // [0,1] -> [-1,1] decode. Standard (OpenGL-style) tangent-space
        // convention: texel (0.5,0.5,1.0) - the "flat", unperturbed resting
        // value - decodes to (0,0,1), i.e. straight along `normal`.
        Vec3 tangentSpaceNormal = texel * 2.0f - Vec3(1.0f);

        // Re-orthonormalize per-hit (Gram-Schmidt) - the barycentric
        // interpolation of 3 per-vertex tangents isn't guaranteed to stay
        // exactly perpendicular to the interpolated normal, even though
        // each contributing per-vertex tangent was orthonormalized at
        // mesh-build time (Mesh::ComputeTangents). Deliberately NOT
        // BasisFromVector (math/Geometric.h) - that builds an arbitrary
        // frame for BSDF importance sampling, not a UV-aligned one, and
        // would rotate the normal map's up-axis randomly per-hit.
        Vec3 t = SafeNormalize(tangent - normal * Dot(normal, tangent), tangent);
        Vec3 b = Cross(normal, t);

        Vec3 perturbed = SafeNormalize(
            t * tangentSpaceNormal.x + b * tangentSpaceNormal.y + normal * tangentSpaceNormal.z, normal);

        // Safety net against a normal map strong enough to flip the
        // perturbed normal onto the opposite side of the (already
        // correctly face-forwarded) unperturbed shading normal - falls
        // back to the unperturbed normal rather than a reflected/clamped
        // variant, the simplest fix that can't introduce a new artifact of
        // its own. Checked against the smooth shading normal, not the raw
        // geometric triangle normal (not available at this call site,
        // which only sees the already-interpolated PrimitiveIntersect
        // output) - a deliberate v1 simplification; the fine-grained
        // distinction between the two only matters on very low-poly,
        // heavily-smoothed meshes.
        if (Dot(perturbed, normal) <= 0.0f)
            return normal;

        return perturbed;
    }
}
