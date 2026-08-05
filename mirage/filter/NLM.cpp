#include "mirage/filter/NLM.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Mirage
{
    namespace
    {
        void AverageFilter(const Color *in, Color *out, int width, int height, int radius)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    int xlower = std::max(0, x - radius);
                    int xupper = std::min(width - 1, x + radius);

                    int ylower = std::max(0, y - radius);
                    int yupper = std::min(height - 1, y + radius);

                    int count = 0;
                    Color sum;

                    for (int fx = xlower; fx <= xupper; ++fx)
                    {
                        for (int fy = ylower; fy <= yupper; ++fy)
                        {
                            sum += in[fy * width + fx];
                            count += 1;
                        }
                    }

                    out[y * width + x] = sum * (1.0f / count);
                }
            }
        }

        // Guide buffers (albedo/normal) use their own, fixed falloff rather
        // than the caller-supplied radiance `falloff` - they're a different
        // quantity on a different natural scale (roughly [0,1]-normalized
        // colors/directions, not open-ended HDR radiance), so reusing the
        // radiance falloff directly would make the guide term's strength an
        // accident of whatever value the caller picked for radiance
        // denoising. Tuned to be fairly strict (small albedo/normal
        // differences already suppress a neighbor's contribution
        // significantly) since the whole point is edge preservation.
        constexpr float kGuideFalloff = 64.0f;
    } // namespace

    void NonLocalMeansFilter(const Color *in, Color *out, int width, int height, float falloff, int radius,
                              const Color *guideAlbedo, const Color *guideNormal)
    {
        const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::vector<Color> means(pixelCount);

        AverageFilter(in, means.data(), width, height, radius);

        float invRadiusSq = falloff; // 200.0f;//1.0f/(0.5f*0.5f);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int xlower = std::max(0, x - radius);
                int xupper = std::min(width - 1, x + radius);

                int ylower = std::max(0, y - radius);
                int yupper = std::min(height - 1, y + radius);

                float totalWeight = 0.0f;
                Color sum;

                int centerIdx = y * width + x;
                Color mean = means[centerIdx];

                for (int fx = xlower; fx <= xupper; ++fx)
                {
                    for (int fy = ylower; fy <= yupper; ++fy)
                    {
                        int neighborIdx = fy * width + fx;

                        float weight = expf(-invRadiusSq * LengthSq(mean - means[neighborIdx]));
                        if (guideAlbedo)
                            weight *= expf(-kGuideFalloff * LengthSq(guideAlbedo[centerIdx] - guideAlbedo[neighborIdx]));
                        if (guideNormal)
                            weight *= expf(-kGuideFalloff * LengthSq(guideNormal[centerIdx] - guideNormal[neighborIdx]));

                        sum += in[neighborIdx] * weight;
                        totalWeight += weight;
                    }
                }

                out[centerIdx] = sum * (1.0f / totalWeight);
            }
        }
    }
}
