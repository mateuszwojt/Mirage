#pragma once

#include "mirage/math/Color.h"

namespace Mirage
{
    // Non-Local-Means denoise: for each pixel, re-weights every neighbor
    // within `radius` by how similar its local mean radiance is (governed by
    // `falloff` - larger falloff means only very similar neighbors
    // contribute), then outputs the weighted average. `in`/`out` may not
    // alias; both must point to `width*height` Color arrays.
    //
    // guideAlbedo/guideNormal (optional, both default nullptr): cross-
    // bilateral guide buffers, the same idea production denoisers (Intel
    // OIDN, NVIDIA's OptiX AI denoiser) use albedo/normal for, just a much
    // simpler hand-rolled version here. Plain radiance-only NLM can mistake
    // legitimate high-frequency detail (a busy texture, a sharp material
    // edge) for noise and blur across it; multiplying in an extra Gaussian
    // similarity term per guide buffer means a neighbor whose albedo/normal
    // differs from the center pixel's contributes less, even if its
    // (possibly still-noisy) radiance happens to look similar - helping the
    // filter tell "this differs because it's noise" apart from "this
    // differs because it's a different surface." Passing nullptr for both
    // (the default) reduces exactly to the original radiance-only behavior.
    // If provided, guide buffers must also be `width*height` Color arrays,
    // same indexing as `in`/`out` - typically AovBuffers::albedo/normal from
    // the same Render() call being denoised.
    void NonLocalMeansFilter(const Color *in, Color *out, int width, int height, float falloff, int radius,
                              const Color *guideAlbedo = nullptr, const Color *guideNormal = nullptr);
}
