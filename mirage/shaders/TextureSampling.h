#pragma once

// Host-only (no CUDA_CALLABLE - runs on the CPU renderer only, mirroring
// TextureLoader.h/Skylight.h's "host-only, not a GPU kernel" precedent).
// Bilinear, wrap-addressed sampling of a linear-space RGBA Texture (see
// Texture.h for the data layout) - deliberately matches the GPU's hardware
// Sampler2D bilinear+wrap sampling (Material.slang, Material_SampleAlbedo
// et al.) rather than the CPU's historical nearest-texel probe-sampling
// convention (core/Probe.h's SampleProbeSphere/ProbeEval), since texture
// sampling is new on both sides simultaneously here - there is no legacy CPU
// behavior to diverge from, so kernel_validate can hold this to a tight
// CPU-vs-GPU bar instead of the probe's pre-approved looser one.

#include "mirage/shaders/Texture.h"
#include "mirage/math/Vec2.h"
#include "mirage/math/Vec3.h"
#include "mirage/math/Vec4.h"

#include <cmath>

namespace Mirage
{
    namespace detail
    {
        inline int WrapCoord(int x, int size)
        {
            int m = x % size;
            return m < 0 ? m + size : m;
        }

        inline Vec4 FetchTexel(const Texture &tex, int x, int y)
        {
            int idx = (WrapCoord(y, tex.height) * tex.width + WrapCoord(x, tex.width)) * 4;
            return Vec4(tex.data[idx + 0], tex.data[idx + 1], tex.data[idx + 2], tex.data[idx + 3]);
        }

        inline Vec4 SampleBilinear(const Texture &tex, Vec2 uv)
        {
            // half-texel-centered bilinear, wrap-addressed on both axes -
            // matches TextureAddressingMode::Wrap used for the GPU material
            // sampler (VulkanRenderer.cpp's UploadMaterialTextures, M8).
            float fx = uv.x * tex.width - 0.5f;
            float fy = uv.y * tex.height - 0.5f;

            int x0 = (int)floorf(fx);
            int y0 = (int)floorf(fy);
            float tx = fx - (float)x0;
            float ty = fy - (float)y0;

            Vec4 c00 = FetchTexel(tex, x0, y0);
            Vec4 c10 = FetchTexel(tex, x0 + 1, y0);
            Vec4 c01 = FetchTexel(tex, x0, y0 + 1);
            Vec4 c11 = FetchTexel(tex, x0 + 1, y0 + 1);

            Vec4 c0 = c00 * (1.0f - tx) + c10 * tx;
            Vec4 c1 = c01 * (1.0f - tx) + c11 * tx;
            return c0 * (1.0f - ty) + c1 * ty;
        }
    }

    // RGB of a bilinear sample - used for albedo (Material::color override).
    inline Vec3 SampleTextureRGB(const Texture &tex, Vec2 uv)
    {
        if (!tex.data || tex.width <= 0 || tex.height <= 0)
            return Vec3(0.0f);
        Vec4 c = detail::SampleBilinear(tex, uv);
        return Vec3(c.x, c.y, c.z);
    }

    // Red channel of a bilinear sample - used for roughness/metallic
    // (single-channel data textures; .r is the conventional single-output
    // component for a scalar-valued UsdUVTexture, see the plan's M10 section).
    inline float SampleTextureR(const Texture &tex, Vec2 uv)
    {
        if (!tex.data || tex.width <= 0 || tex.height <= 0)
            return 0.0f;
        return detail::SampleBilinear(tex, uv).x;
    }

    // Alpha channel of a bilinear sample - used for opacity/cutout
    // transparency (Material::opacity override). Alpha is already loaded
    // into Texture::data's 4th channel by TextureLoader.cpp (and already
    // uploaded to GPU as Format::RGBA32Float) but was never sampled
    // anywhere until this. Falls back to 1.0 (fully opaque) rather than 0.0
    // on missing/invalid texture data - an absent texture should never
    // silently make geometry disappear.
    inline float SampleTextureAlpha(const Texture &tex, Vec2 uv)
    {
        if (!tex.data || tex.width <= 0 || tex.height <= 0)
            return 1.0f;
        return detail::SampleBilinear(tex, uv).w;
    }
}
