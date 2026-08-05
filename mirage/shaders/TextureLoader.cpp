#include "mirage/shaders/TextureLoader.h"
#include "mirage/math/Color.h"
#include "mirage/utils/Log.h"

#define STB_IMAGE_IMPLEMENTATION
#include "mirage/thirdparty/stb_image.h"

#include <cmath>

namespace Mirage
{
    std::unique_ptr<Texture> LoadTextureFromFile(const std::string &path, TextureColorSpace colorSpace, bool generateMips)
    {
        int width = 0, height = 0, channelsInFile = 0;
        // Request 4 channels (RGBA) unconditionally - Texture::data's documented
        // layout is always RGBA-interleaved (see Texture.h), regardless of how
        // many channels the source file actually has (stb_image pads/replicates
        // as needed: grayscale -> replicated into R/G/B, no alpha -> alpha = 1.0).
        float *pixels = stbi_loadf(path.c_str(), &width, &height, &channelsInFile, 4);
        if (!pixels)
        {
            MIRAGE_LOG_ERROR("LoadTextureFromFile: failed to load '{}': {}", path, stbi_failure_reason());
            return nullptr;
        }

        auto tex = std::make_unique<Texture>();
        tex->width = width;
        tex->height = height;
        tex->depth = 1;

        size_t count = (size_t)width * (size_t)height * 4;
        tex->data = new float[count];

        // stbi_loadf normalizes 8-bit LDR sources to [0,1] via a plain /255
        // divide (no gamma handling), so an sRGB->linear decode is needed for
        // LDR sources tagged eSRGB. True HDR sources (.hdr/.pic) are already
        // linear on disk - stbi_is_hdr() distinguishes the two so a caller
        // that (correctly) tags an albedo texture eSRGB doesn't get a bogus
        // second gamma decode applied on top of an already-linear .hdr file.
        bool applySrgbDecode = colorSpace == TextureColorSpace::eSRGB && !stbi_is_hdr(path.c_str());
        if (applySrgbDecode)
        {
            for (size_t i = 0; i < count; i += 4)
            {
                tex->data[i + 0] = SrgbToLinear(pixels[i + 0]);
                tex->data[i + 1] = SrgbToLinear(pixels[i + 1]);
                tex->data[i + 2] = SrgbToLinear(pixels[i + 2]);
                tex->data[i + 3] = pixels[i + 3];
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
                tex->data[i] = pixels[i];
        }

        stbi_image_free(pixels);

        if (generateMips)
            tex->GenerateMips();

        return tex;
    }
}
