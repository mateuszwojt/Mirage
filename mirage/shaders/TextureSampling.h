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
#include "mirage/utils/MathUtils.h"

#include <cmath>

namespace Mirage
{
    // UDIM atlas UV remap (Tier-2 "UDIM"): a no-op (returns uv unchanged) for
    // an ordinary single-tile texture (udimGridWidth == udimGridHeight == 1,
    // the default). For a UDIM atlas, `uv`'s integer part selects a tile
    // (tile (0,0) = the atlas's top-left cell in memory - u increasing
    // right, v increasing down the row-major grid, matching this codebase's
    // existing uv.y=0-is-the-first-stored-row sampling convention exactly,
    // so no vertical flip is needed between the loader that builds the
    // atlas - LoadUdimAtlas in SceneRenderer.cpp - and this remap) and the
    // fractional part is the local within-tile UV; both are remapped into
    // the composited atlas texture's own [0,1] UV space. Kept separate from
    // the mip/bilinear sampling below - it's a pure UV transform, applied
    // once, before any of it.
    inline Vec2 UdimAtlasUV(const Texture &tex, Vec2 uv)
    {
        if (tex.udimGridWidth <= 1 && tex.udimGridHeight <= 1)
            return uv;

        float tileU = floorf(uv.x);
        float tileV = floorf(uv.y);
        float localU = uv.x - tileU;
        float localV = uv.y - tileV;

        // Out-of-range tile indices (a UV outside the authored tile set)
        // clamp to the nearest edge tile rather than wrapping - wrapping
        // would silently sample an unrelated tile's content, which is a much
        // more confusing failure mode than "the edge tile repeats."
        float clampedU = ::Mirage::Clamp(tileU, 0.0f, float(tex.udimGridWidth - 1));
        float clampedV = ::Mirage::Clamp(tileV, 0.0f, float(tex.udimGridHeight - 1));

        return Vec2((clampedU + localU) / float(tex.udimGridWidth),
                    (clampedV + localV) / float(tex.udimGridHeight));
    }

    namespace detail
    {
        inline int WrapCoord(int x, int size)
        {
            int m = x % size;
            return m < 0 ? m + size : m;
        }

        inline Vec4 FetchTexelLevel(const float *data, int w, int h, int x, int y)
        {
            int idx = (WrapCoord(y, h) * w + WrapCoord(x, w)) * 4;
            return Vec4(data[idx + 0], data[idx + 1], data[idx + 2], data[idx + 3]);
        }

        inline Vec4 SampleBilinearLevel(const float *data, int w, int h, Vec2 uv)
        {
            // half-texel-centered bilinear, wrap-addressed on both axes -
            // matches TextureAddressingMode::Wrap used for the GPU material
            // sampler (VulkanRenderer.cpp's UploadMaterialTextures, M8).
            float fx = uv.x * w - 0.5f;
            float fy = uv.y * h - 0.5f;

            int x0 = (int)floorf(fx);
            int y0 = (int)floorf(fy);
            float tx = fx - (float)x0;
            float ty = fy - (float)y0;

            Vec4 c00 = FetchTexelLevel(data, w, h, x0, y0);
            Vec4 c10 = FetchTexelLevel(data, w, h, x0 + 1, y0);
            Vec4 c01 = FetchTexelLevel(data, w, h, x0, y0 + 1);
            Vec4 c11 = FetchTexelLevel(data, w, h, x0 + 1, y0 + 1);

            Vec4 c0 = c00 * (1.0f - tx) + c10 * tx;
            Vec4 c1 = c01 * (1.0f - tx) + c11 * tx;
            return c0 * (1.0f - ty) + c1 * ty;
        }

        // Trilinear: bilinear-samples the two integer mip levels bracketing
        // `mipLevel` and blends between them. Falls back to a single
        // bilinear sample of level 0 (tex.data/width/height, NOT
        // tex.mipData[0] - identical content, but avoids depending on
        // GenerateMips() ever having run) whenever no mip chain exists
        // (tex.mipCount <= 1) or the caller asks for level 0 exactly - this
        // is the path every pre-existing (pre-mipmapping) call site takes,
        // unchanged.
        inline Vec4 SampleTrilinear(const Texture &tex, Vec2 uv, float mipLevel)
        {
            if (tex.mipCount <= 1 || mipLevel <= 0.0f || !tex.mipData)
                return SampleBilinearLevel(tex.data, tex.width, tex.height, uv);

            float clampedLevel = ::Mirage::Clamp(mipLevel, 0.0f, float(tex.mipCount - 1));
            int level0 = (int)floorf(clampedLevel);
            int level1 = ::Mirage::Min(level0 + 1, tex.mipCount - 1);
            float t = clampedLevel - float(level0);

            Vec4 c0 = SampleBilinearLevel(tex.mipData[level0], tex.mipWidths[level0], tex.mipHeights[level0], uv);
            if (level0 == level1)
                return c0;

            Vec4 c1 = SampleBilinearLevel(tex.mipData[level1], tex.mipWidths[level1], tex.mipHeights[level1], uv);
            return c0 * (1.0f - t) + c1 * t;
        }
    }

    // RGB of a (tri)linear sample - used for albedo (Material::color
    // override). mipLevel defaults to 0 (base level only, no mip blending) -
    // callers that want mip filtering pass an explicit level (see
    // Renderer.cpp's PathTrace() mip-level heuristic).
    inline Vec3 SampleTextureRGB(const Texture &tex, Vec2 uv, float mipLevel = 0.0f)
    {
        if (!tex.data || tex.width <= 0 || tex.height <= 0)
            return Vec3(0.0f);
        Vec4 c = detail::SampleTrilinear(tex, UdimAtlasUV(tex, uv), mipLevel);
        return Vec3(c.x, c.y, c.z);
    }

    // Red channel of a (tri)linear sample - used for roughness/metallic
    // (single-channel data textures; .r is the conventional single-output
    // component for a scalar-valued UsdUVTexture, see the plan's M10 section).
    inline float SampleTextureR(const Texture &tex, Vec2 uv, float mipLevel = 0.0f)
    {
        if (!tex.data || tex.width <= 0 || tex.height <= 0)
            return 0.0f;
        return detail::SampleTrilinear(tex, UdimAtlasUV(tex, uv), mipLevel).x;
    }

    // Alpha channel of a (tri)linear sample - used for opacity/cutout
    // transparency (Material::opacity override). Alpha is already loaded
    // into Texture::data's 4th channel by TextureLoader.cpp (and already
    // uploaded to GPU as Format::RGBA32Float) but was never sampled
    // anywhere until this. Falls back to 1.0 (fully opaque) rather than 0.0
    // on missing/invalid texture data - an absent texture should never
    // silently make geometry disappear.
    inline float SampleTextureAlpha(const Texture &tex, Vec2 uv, float mipLevel = 0.0f)
    {
        if (!tex.data || tex.width <= 0 || tex.height <= 0)
            return 1.0f;
        return detail::SampleTrilinear(tex, UdimAtlasUV(tex, uv), mipLevel).w;
    }
}
