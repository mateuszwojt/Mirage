#pragma once

#include "mirage/core/Renderer.h"
#include "mirage/kernels/GpuLayout.h"

namespace rhi
{
    class IDevice;
}

namespace Mirage
{
    // GPU renderer backed by Slang-compiled compute kernels running through
    // slang-rhi's Vulkan backend (MoltenVK on macOS). This is the only
    // GPU-capable render path going forward - see
    // ~/.claude/plans/think-through-refactoring-mirage-fluttering-rainbow.md.
    //
    // M0: device creation + a trivial validation kernel (ClearColor.slang).
    // M1: scene upload (primitives/meshes/BVH) + eNormals-mode dispatch
    // (NormalsMain.slang).
    // M2: Disney BSDF port, validated via RunBsdfTest() (isolated unit-test
    // kernel, no scene/BVH/camera involved).
    // M3: full path integrator (Trace/SampleLights/PathTrace/AddSample) +
    // RenderMode::ePathTrace dispatch via PathTraceMain.slang, with progressive
    // box-filter accumulation. Validated primarily via RunPathTraceTest()
    // (isolated, explicitly-seeded unit-test kernel - see its header comment for
    // why per-pixel-dispatch RNG makes a tight full-image comparison the wrong
    // tool); the real per-pixel Render(ePathTrace) path is exercised for
    // "runs and looks plausible" but not held to a strict CPU-vs-GPU MSE gate.
    // M4: HDR environment probe support (sampling/Probe.slang) - importance
    // sampling, MIS-weighted evaluation on a miss, and NEE sky sampling, wired
    // into both PathTrace.slang and SampleLights.slang. UploadScene() always
    // uploads a probe texture/CDF/PDF buffers (a 1x1 dummy when the scene has
    // none), since the pipeline's descriptor layout is fixed regardless of the
    // runtime probeValid branch. Validated via RunProbeTest().
    struct VulkanRenderer : public Renderer
    {
        VulkanRenderer(const Scene *s);
        ~VulkanRenderer() override;

        void Init(int width, int height) override;
        void Render(const Camera &c, const Options &options, Color *output, AovBuffers *aovs = nullptr) override;

        // Re-zeroes the progressive accumulation buffer in place, without
        // re-uploading scene geometry - much cheaper than recreating the
        // renderer. Call when the camera changes between Render() calls.
        void ResetAccumulation() override;

        // True if a Vulkan device was created successfully. Callers (e.g.
        // HydraCompatibility) should fall back to CreateCpuRenderer() if false.
        bool IsAvailable() const;

        // M2 validation-only: runs BsdfTestMain.slang over `count` independent
        // test cases (no scene/BVH/camera involved) - the GPU side of
        // kernel_validate's BSDF Eval/Pdf/Sample comparison against
        // mirage/mirage/shaders/Disney.h.
        void RunBsdfTest(const GpuBsdfTestCase *cases, GpuBsdfTestResult *results, int count);

        // M3 validation-only: runs PathTraceTestMain.slang over `count`
        // independent, explicitly-seeded primary rays through the uploaded
        // scene - the GPU side of kernel_validate's PathTrace comparison
        // against mirage/mirage/core/Renderer.cpp's PathTrace().
        void RunPathTraceTest(const GpuPathTraceTestCase *cases, GpuPathTraceTestResult *results, int count);

        // M4 validation-only: runs ProbeTestMain.slang over `count` independent
        // test cases (no scene BVH/materials/camera involved, just the probe
        // texture/CDF/PDF buffers) - the GPU side of kernel_validate's
        // ProbeSample/ProbeEval/ProbePdf comparison against
        // mirage/mirage/core/Probe.h.
        void RunProbeTest(const GpuProbeTestCase *cases, GpuProbeTestResult *results, int count);

        // M5 validation-only: runs TextureArraySpike.slang, which samples 4
        // distinct textures bound through a Sampler2D g_Textures[4] array via
        // slang-rhi's ShaderCursor - the go/no-go spike for the array-of-
        // textures binding mechanism M8's real per-material texture array
        // depends on. Returns false if the device/pipeline isn't available.
        static const uint32_t kTextureArraySpikeCount = 4;
        bool RunTextureArraySpike(GpuTextureArraySpikeResult (&outColors)[kTextureArraySpikeCount]);

        // M8 validation-only: runs TextureSampleTestMain.slang over `count`
        // independent (material, uv) cases - the GPU side of kernel_validate's
        // Material_SampleAlbedo/Roughness/Metallic comparison against
        // TextureSampling.h's CPU implementation. `materials` indexes into
        // *scene's textures via its xxxTextureIndex fields; no primitives/BVH/
        // camera are involved.
        bool RunTextureSampleTest(const GpuMaterial *materials, int materialCount,
                                   const GpuTextureSampleTestCase *cases, GpuTextureSampleTestResult *results, int count);

    private:
        struct Impl;
        Impl *impl;
    };
}
