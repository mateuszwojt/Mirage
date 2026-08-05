#pragma once

#include "mirage/shaders/Texture.h"

#include <memory>
#include <string>

namespace Mirage
{
    // Which transfer function to apply while decoding an on-disk image into
    // Texture::data. Color textures (albedo) need sRGB->linear decode since
    // Material::color is consumed as a linear reflectance value throughout
    // Disney.h; data textures (roughness/metallic) must NOT be sRGB-decoded -
    // matches UsdPreviewSurface's sourceColorSpace convention (default sRGB
    // for diffuseColor-bound textures, raw for non-color inputs).
    enum class TextureColorSpace
    {
        eSRGB,
        eLinear,
    };

    // Decodes an image file (any format stb_image.h supports: PNG/JPG/BMP/
    // TGA/HDR/...) into a Texture whose `data` is always linear-space RGBA
    // float regardless of the source file's on-disk encoding or bit depth -
    // sampling code never needs to know the original format. Returns nullptr
    // (and does not throw) if the file can't be opened or decoded; callers
    // should treat that as "no texture" (fall back to the material's scalar/
    // color field), not a fatal error - a missing/broken texture reference
    // shouldn't take down a whole render.
    //
    // generateMips (Tier-2 "mipmapping"): on by default - calls
    // Texture::GenerateMips() before returning, so ordinary callers get a
    // full mip chain for free. Callers that don't want the extra memory/CPU
    // cost (e.g. tiny 1x1 test textures where a mip chain is meaningless, or
    // a UDIM tile that's about to be composited into an atlas and mipped
    // once as a whole afterward - see the UDIM atlas loader) can pass false.
    std::unique_ptr<Texture> LoadTextureFromFile(const std::string &path, TextureColorSpace colorSpace, bool generateMips = true);
}
