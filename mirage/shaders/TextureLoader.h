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
    std::unique_ptr<Texture> LoadTextureFromFile(const std::string &path, TextureColorSpace colorSpace);
}
