#pragma once

#include <cstdint>

#include "mirage/utils/MathUtils.h"
#include "mirage/core/Scene.h"
#include "mirage/core/Intersection.h"

namespace Mirage
{
    using ::Mirage::Camera;
    using ::Mirage::Color;
    enum FilterType
    {
        eFilterBox = 0,
        eFilterGaussian = 1
    };

    struct Filter
    {
        CUDA_CALLABLE Filter(FilterType type = eFilterGaussian, float width = 1.0f, float falloff = 2.0f) : type(type), width(width), falloff(falloff)
        {
            if (type == eFilterGaussian)
                offset = expf(-falloff * width * width);
        }

        CUDA_CALLABLE float Eval(float x, float y) const
        {
            if (type == eFilterGaussian)
                return Gaussian(x) * Gaussian(y);
            else
                return 1.0f;
        }

        CUDA_CALLABLE float Gaussian(float x) const
        {
            return ::Mirage::Max(0.0f, float(expf(-falloff * x * x)) - offset);
        }

        FilterType type;

        float width;
        float falloff;
        float offset;
    };

    enum RenderType
    {
        eCpu = 0,
        eCpuWavefront = 1,
        eGpu = 2,
        eGpuWavefront = 3
    };

    enum RenderMode
    {
        eNormals = 0,
        eComplexity = 1,
        ePathTrace = 2
    };

    // Bitmask values for Options::aovMask - which auxiliary first-hit buffers
    // a Render() call should populate, in addition to the always-computed
    // beauty/color output. All three are single-valued per-pixel snapshots of
    // the primary ray hit (not progressively accumulated like color), see
    // AovBuffers.
    enum AovType : uint32_t
    {
        kAovDepth  = 1u << 0, // .x = primary-ray hit distance; AovBuffers::depth
        kAovNormal = 1u << 1, // .xyz = world-space shading normal; AovBuffers::normal
        kAovPrimId = 1u << 2, // .x = (float)Primitive::hydraId, -1.0f on miss; AovBuffers::primId
    };

    // Caller-owned output buffers for the AOVs requested via
    // Options::aovMask. Each pointer, if non-null, must point to a
    // width*height Color array (matching the primary `output` array's
    // dimensions). A set aovMask bit with a null pointer here is treated as
    // "not requested" by both backends, not a crash.
    struct AovBuffers
    {
        Color *depth = nullptr;
        Color *normal = nullptr;
        Color *primId = nullptr;
    };

    struct Options
    {
        RenderType type;
        RenderMode mode;
        int width;
        int height;

        Filter filter;
        float exposure;
        float limit;
        float clamp;

        float nlmWidth;
        float nlmFalloff;

        int maxDepth;
        int maxSamples;

        bool enableDOF;

        // Bitwise-or of AovType - which auxiliary buffers to populate this
        // call, in addition to color. 0 (the default) means color only,
        // preserving the behavior of every pre-existing call site.
        uint32_t aovMask = 0;
    };

    struct Renderer
    {
        virtual ~Renderer() {}

        virtual void Init(int width, int height) {}
        virtual void Render(const Camera &c, const Options &options, Color *output, AovBuffers *aovs = nullptr) = 0;

        // Clear any persistent progressive-accumulation state (e.g. a GPU
        // accumulation buffer) without discarding uploaded scene data. Callers
        // use this when only the camera moved, so subsequent samples don't
        // blend with pixels traced from a stale camera position. Backends with
        // no such persistent state (the CPU renderer accumulates entirely in
        // the caller-owned output buffer) can leave this a no-op.
        virtual void ResetAccumulation() {}
    };

    Renderer *CreateNullRenderer(const Scene *s);
    Renderer *CreateCpuRenderer(const Scene *s);
    Renderer *CreateCpuWavefrontRenderer(const Scene *s);
    Renderer *CreateGpuWavefrontRenderer(const Scene *s);
    Renderer *CreateVulkanRenderer(const Scene *s);
}
