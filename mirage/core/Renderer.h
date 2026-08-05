#pragma once

#include <cstdint>
#include <string>
#include <vector>

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
    // beauty/color output. By default all four are single-valued per-pixel
    // snapshots of the primary ray hit (not progressively accumulated like
    // color) - see AovBuffers and Options::accumulateAovs to opt into
    // running-mean accumulation instead (CPU backend only today).
    enum AovType : uint32_t
    {
        kAovDepth  = 1u << 0, // .x = primary-ray hit distance; AovBuffers::depth
        kAovNormal = 1u << 1, // .xyz = world-space shading normal; AovBuffers::normal
        kAovPrimId = 1u << 2, // .x = (float)Primitive::hydraId, -1.0f on miss; AovBuffers::primId
        kAovAlbedo = 1u << 3, // .xyz = resolved (texture-sampled) base color at the primary hit, .w unused; AovBuffers::albedo
    };

    // Tier B / "arbitrary" AOVs (docs/PRODUCTION_READINESS.md's Tier 3 "no
    // arbitrary/user-defined AOVs" finding): requested by name rather than a
    // fixed AovType bit, CPU backend only today - the GPU/Vulkan backend
    // leaves every NamedAov::buffer untouched and logs a one-time warning
    // per Render() call if any are requested, matching AovBuffers' existing
    // "not populated" convention rather than crashing. See
    // mirage/core/Renderer.cpp's PathTrace() for the fixed set of names it
    // currently recognizes (e.g. "P" world-space hit position, "uv" surface
    // UV) - an unrecognized name is silently left unpopulated too, the same
    // "caller over-requested, not an error" convention as a set AovType bit
    // with a null AovBuffers pointer.
    struct NamedAov
    {
        std::string name;
        Color *buffer = nullptr;
    };

    // Caller-owned output buffers for the AOVs requested via
    // Options::aovMask (depth/normal/primId/albedo) plus `named` (Tier B,
    // see NamedAov). Each AovType pointer, if non-null, must point to a
    // width*height Color array (matching the primary `output` array's
    // dimensions); same requirement for every NamedAov::buffer in `named`. A
    // set aovMask bit with a null pointer here is treated as "not requested"
    // by both backends, not a crash.
    struct AovBuffers
    {
        Color *depth = nullptr;
        Color *normal = nullptr;
        Color *primId = nullptr;
        Color *albedo = nullptr;

        std::vector<NamedAov> named;
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

        // Non-Local-Means post-process denoise (mirage/filter/NLM.h) config -
        // NOT consumed by Render() itself (no backend denoises its own
        // output); a CLI/pipeline caller reads these after Render() returns
        // and applies NonLocalMeansFilter() to the output buffer directly
        // (see tools/scene_renderer/SceneRenderer.cpp's --denoise flag).
        // Housed on Options anyway (not a SceneRenderer-local variable) so
        // denoise config travels with the rest of a render's settings the
        // same way aovMask/accumulateAovs do - previously declared here but
        // never read anywhere, a dead "the previous implementer got this far
        // and stopped" finding (docs/PRODUCTION_READINESS.md's Tier 3
        // "denoiser is dead code" item).
        float nlmWidth;   // neighborhood radius in pixels (NonLocalMeansFilter's `radius`, rounded to int)
        float nlmFalloff; // similarity weight falloff (NonLocalMeansFilter's `falloff`)

        int maxDepth;
        int maxSamples;

        bool enableDOF;

        // Bitwise-or of AovType - which auxiliary buffers to populate this
        // call, in addition to color. 0 (the default) means color only,
        // preserving the behavior of every pre-existing call site.
        uint32_t aovMask = 0;

        // false (default): AovType buffers are first-hit-only snapshots,
        // preserving the original behavior every pre-existing call site
        // (HydraCompatibility.cpp/HdMirageRenderer) relies on - captured
        // once, on the primary-ray hit of the *first* sample only, never
        // overwritten by later samples' antialiasing jitter. true: instead
        // accumulate a running per-pixel mean across all `maxSamples`
        // samples, the same way the beauty buffer progressively converges -
        // CPU backend only today; the GPU/Vulkan backend ignores this flag
        // and keeps first-sample-only behavior regardless (its AOV buffers
        // are already overwritten-not-accumulated by construction - see
        // VulkanRenderer.cpp).
        bool accumulateAovs = false;

        // false (default): no post-process denoise, preserving every
        // pre-existing call site's behavior. true: the CLI/pipeline caller
        // (not Render() itself - see nlmWidth/nlmFalloff above) applies
        // NonLocalMeansFilter to the beauty buffer using nlmWidth/nlmFalloff.
        bool enableDenoise = false;
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
