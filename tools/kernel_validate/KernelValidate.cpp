// Standalone (non-Hydra) validation harness - the fast local iteration/
// correctness loop for the Slang/Vulkan kernel port, per the plan
// (~/.claude/plans/think-through-refactoring-mirage-fluttering-rainbow.md).
//
// M1: builds a scene with one sphere + one mesh primitive, renders it with both
// CreateCpuRenderer and CreateVulkanRenderer in eNormals mode, and reports a
// per-pixel MSE. eNormals writes a final Color directly (no filter-splat
// accumulation/tonemap involved), so no normalization step is needed there.
//
// M2: RunBsdfValidation() evaluates BSDFEval/BSDFPdf/BSDFSample for a batch of
// independent (material, N, V, L, seed) test cases directly - no scene/BVH/
// camera involved - comparing mirage/mirage/shaders/Disney.h's CPU
// implementation against kernels/slang/shading/Disney.slang on the GPU.
//
// M3: RunPathTraceValidation() calls PathTrace() directly with explicitly
// seeded, independent primary rays through a small scene (sphere + mesh +
// emissive-sphere light), comparing against kernels/slang/integrator/
// PathTrace.slang on the GPU.
//
// M4: RunProbeValidation() evaluates ProbeSample/ProbeEval/ProbePdf for a
// batch of independent (seed, direction) test cases against a small synthetic
// HDR environment probe, comparing mirage/mirage/core/Probe.h's CPU
// implementation against kernels/slang/sampling/Probe.slang on the GPU.

#include "mirage/core/Renderer.h"
#include "mirage/core/VulkanRenderer.h"
#include "mirage/core/Scene.h"
#include "mirage/camera/Camera.h"
#include "mirage/prims/Mesh.h"
#include "mirage/utils/Util.h"
#include "mirage/shaders/Disney.h"
#include "mirage/shaders/TextureSampling.h"
#include "mirage/shaders/NormalMapping.h"
#include "mirage/math/Geometric.h"
#include "mirage/core/Probe.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

using namespace Mirage;

namespace Mirage
{
    // PathTrace() is defined (non-inline, so it has a stable external symbol)
    // in mirage/mirage/core/Renderer.cpp but never declared in Renderer.h -
    // it's an internal implementation detail of CpuRenderer, not part of the
    // public Renderer API. Declared here to link directly against it for the
    // M3 CPU-vs-GPU comparison, exactly matching Renderer.cpp's signature
    // (including the trailing AOV out-params added for depth/normal/primId
    // capture - default arguments are resolved at the call site, but the
    // mangled symbol still reflects the full parameter list, so this
    // declaration must match exactly or the link fails).
    Vec3 PathTrace(const Scene &scene, const Vec3 &startOrigin, const Vec3 &startDir, float time, int maxDepth, Random &rand,
                   float *outDepth = nullptr, Vec3 *outNormal = nullptr, const Primitive **outHitPrim = nullptr);
}

namespace
{
    std::unique_ptr<Mesh> MakeQuadMesh()
    {
        auto mesh = std::make_unique<Mesh>();
        mesh->vertices = {
            Vec3(-0.7f, -0.7f, 0.0f),
            Vec3(0.7f, -0.7f, 0.0f),
            Vec3(0.7f, 0.7f, 0.0f),
            Vec3(-0.7f, 0.7f, 0.0f),
        };
        mesh->indices = {0, 1, 2, 0, 2, 3};
        mesh->CalculateNormals();
        mesh->rebuildBVH();
        return mesh;
    }

    Camera BuildTestCamera()
    {
        Camera camera;
        camera.position = Vec3(0.0f, 0.0f, 6.0f);
        camera.fov = DegToRad(50.0f);
        camera.aperture = 0.0f;
        camera.focalPoint = 0.0f;
        return camera;
    }

    // Places test geometry from the camera's *actual* rays rather than
    // hand-derived world-space coordinates. CameraSampler's raster->world
    // composition (mirage/mirage/utils/Util.h) does not produce the naively
    // expected symmetric frustum for this camera/resolution combination - the
    // visible window at a given depth is a real, but off-center and much
    // narrower than fov-implied, quadrilateral. Rather than chase that down
    // further (a pre-existing CPU-side quirk, unrelated to the Slang/Vulkan
    // port - GPU has already been confirmed to reproduce it exactly), derive
    // placement from CameraSampler's own center/corner rays so the test scene
    // is guaranteed to be in view regardless.
    void BuildTestScene(Scene &scene, const Camera &camera, int width, int height)
    {
        CameraSampler sampler(Transform(camera.position, camera.rotation), camera.fov, 0.001f, 1.0f,
                               camera.aperture, camera.focalPoint, false, width, height);

        Vec3 origin, centerDir, cornerDir;
        sampler.GenerateRay(width / 2.0f, height / 2.0f, origin, centerDir);
        sampler.GenerateRay(0.0f, 0.0f, origin, cornerDir);

        auto hitZPlane = [&](const Vec3 &dir) {
            float t = -origin.z / dir.z;
            return origin + dir * t;
        };

        Vec3 centerPoint = hitZPlane(centerDir);
        Vec3 cornerPoint = hitZPlane(cornerDir);
        Vec3 spread = cornerPoint - centerPoint;
        float halfWindow = Length(spread);
        Vec3 spreadDir = spread / halfWindow;

        float objRadius = halfWindow * 0.25f;
        Vec3 spherePos = centerPoint - spreadDir * (halfWindow * 0.4f);
        Vec3 meshPos = centerPoint + spreadDir * (halfWindow * 0.4f);

        Primitive spherePrim;
        spherePrim.type = eSphere;
        spherePrim.sphere.radius = 1.0f;
        spherePrim.startTransform = Transform(spherePos, Quat(), objRadius);
        spherePrim.endTransform = spherePrim.startTransform;
        scene.AddPrimitive(spherePrim);

        std::unique_ptr<Mesh> mesh = MakeQuadMesh();
        Primitive meshPrim;
        meshPrim.type = eMesh;
        meshPrim.startTransform = Transform(meshPos, Quat(), objRadius);
        meshPrim.endTransform = meshPrim.startTransform;
        meshPrim.mesh = GeometryFromMesh(mesh.get());
        meshPrim.mesh.meshIndex = scene.AddMesh(std::move(mesh));
        scene.AddPrimitive(meshPrim);

        scene.Build();
    }

    void WritePng(const char *path, int width, int height, const std::vector<Color> &image)
    {
        std::vector<unsigned char> bytes(width * height * 3);
        for (int i = 0; i < width * height; ++i)
        {
            bytes[i * 3 + 0] = (unsigned char)(255.0f * Clamp(image[i].x, 0.0f, 1.0f));
            bytes[i * 3 + 1] = (unsigned char)(255.0f * Clamp(image[i].y, 0.0f, 1.0f));
            bytes[i * 3 + 2] = (unsigned char)(255.0f * Clamp(image[i].z, 0.0f, 1.0f));
        }
        stbi_write_png(path, width, height, 3, bytes.data(), width * 3);
    }

    void WriteDiffHeatmap(const char *path, int width, int height, const std::vector<Color> &a, const std::vector<Color> &b)
    {
        std::vector<unsigned char> bytes(width * height * 3);
        for (int i = 0; i < width * height; ++i)
        {
            float dr = fabsf(a[i].x - b[i].x);
            float dg = fabsf(a[i].y - b[i].y);
            float db = fabsf(a[i].z - b[i].z);
            float mag = Clamp((dr + dg + db) / 3.0f * 8.0f, 0.0f, 1.0f); // amplify for visibility
            bytes[i * 3 + 0] = (unsigned char)(255.0f * mag);
            bytes[i * 3 + 1] = 0;
            bytes[i * 3 + 2] = (unsigned char)(255.0f * (1.0f - mag));
        }
        stbi_write_png(path, width, height, 3, bytes.data(), width * 3);
    }
}

namespace
{
    bool RunNormalsValidation()
    {
        const int width = 256;
        const int height = 256;

        Camera camera = BuildTestCamera();
        Scene scene;
        BuildTestScene(scene, camera, width, height);

        Options options{};
        options.mode = eNormals;
        options.width = width;
        options.height = height;
        options.enableDOF = false;
        options.maxDepth = 1;
        options.maxSamples = 1;
        options.exposure = 1.0f;
        options.limit = 1000.0f;
        options.clamp = 1000.0f;

        std::vector<Color> cpuImage(width * height, Color(0.0f));
        {
            std::unique_ptr<Renderer> cpuRenderer(CreateCpuRenderer(&scene));
            cpuRenderer->Init(width, height);
            cpuRenderer->Render(camera, options, cpuImage.data());
        }

        std::vector<Color> gpuImage(width * height, Color(0.0f));
        {
            std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
            if (!gpuRenderer->IsAvailable())
            {
                fprintf(stderr, "FAIL: VulkanRenderer not available on this machine\n");
                return false;
            }
            gpuRenderer->Init(width, height);
            gpuRenderer->Render(camera, options, gpuImage.data());
        }

        double sumSqError = 0.0;
        int badPixels = 0;
        const float kPerPixelTolerance = 0.05f; // generous for M1: hit/miss boundary AA differences expected

        for (int i = 0; i < width * height; ++i)
        {
            float dr = cpuImage[i].x - gpuImage[i].x;
            float dg = cpuImage[i].y - gpuImage[i].y;
            float db = cpuImage[i].z - gpuImage[i].z;
            float sq = dr * dr + dg * dg + db * db;
            sumSqError += sq;
            if (sq > kPerPixelTolerance * kPerPixelTolerance)
                ++badPixels;
        }

        double mse = sumSqError / (width * height * 3);

        WritePng("kernel_validate_cpu.png", width, height, cpuImage);
        WritePng("kernel_validate_gpu.png", width, height, gpuImage);
        WriteDiffHeatmap("kernel_validate_diff.png", width, height, cpuImage, gpuImage);

        printf("[M1 geometry/BVH] MSE = %f, bad pixels (> %.2f per-channel-ish) = %d / %d\n", mse,
               kPerPixelTolerance, badPixels, width * height);

        // Generous M1 threshold - tightened once geometry/BVH is the only variable
        // left (shading/lighting isn't in the picture yet for eNormals).
        const double kMseThreshold = 0.01;
        const int kBadPixelThreshold = (width * height) / 200; // allow ~0.5% edge-AA mismatch

        bool pass = mse < kMseThreshold && badPixels < kBadPixelThreshold;
        printf("[M1 geometry/BVH] %s\n", pass ? "PASS" : "FAIL");
        return pass;
    }

    // One BSDF Eval/Pdf/Sample test case, generated identically for both the CPU
    // direct-call path and the GPU upload path so the two are apples-to-apples.
    struct BsdfTestInput
    {
        Material material;
        float etaI, etaO;
        Vec3 n, v, l;
        uint32_t seed;
    };

    BsdfTestInput GenerateBsdfTestCase(int ix, int iy, int gridW, int gridH)
    {
        BsdfTestInput in{};
        in.material.color = Vec3(0.8f, 0.5f, 0.3f);
        in.material.emission = Vec3(0.0f);
        in.material.absorption = Vec3(0.0f);
        in.material.eta = 0.0f; // inferred from specular, matches Material()'s default
        in.material.metallic = float(ix % 8) / 7.0f;
        in.material.roughness = Clamp(float((ix / 8) % 8) / 7.0f, 0.03f, 1.0f);
        in.material.specular = 0.5f;
        in.material.specularTint = float((ix + iy) % 3) / 2.0f;
        in.material.anisotropic = 0.0f;
        in.material.sheen = 0.0f;
        in.material.sheenTint = 0.0f;
        in.material.transmission = (float(iy % 4) / 3.0f) * 0.8f;
        in.material.subsurface = (float((iy / 4) % 4) / 3.0f) * 0.5f;
        in.material.clearcoat = (float((ix * 7 + iy * 13) % 5) / 4.0f) * 0.5f;
        in.material.clearcoatGloss = 0.7f;

        in.etaI = 1.0f;
        in.etaO = in.material.GetIndexOfRefraction();

        in.n = Vec3(0.0f, 0.0f, 1.0f);

        // V sweeps grazing->normal incidence across ix; L sweeps a full polar
        // range (including NDotL <= 0, to exercise the transmission/subsurface
        // branches) across iy, at a fixed, non-degenerate azimuthal offset.
        float vTheta = DegToRad(5.0f + 75.0f * (float(ix) / float(gridW - 1)));
        in.v = Normalize(Vec3(sinf(vTheta), 0.0f, cosf(vTheta)));

        float lTheta = DegToRad(-170.0f + 340.0f * (float(iy) / float(gridH - 1)));
        in.l = Normalize(Vec3(sinf(lTheta) * cosf(0.7f), sinf(lTheta) * sinf(0.7f), cosf(lTheta)));

        in.seed = (uint32_t)((ix * gridH + iy) * 2654435761u);
        return in;
    }

    GpuMaterial ToGpuMaterial(const Material &m)
    {
        GpuMaterial g{};
        g.colorX = m.color.x; g.colorY = m.color.y; g.colorZ = m.color.z;
        g.emissionX = m.emission.x; g.emissionY = m.emission.y; g.emissionZ = m.emission.z;
        g.absorptionX = m.absorption.x; g.absorptionY = m.absorption.y; g.absorptionZ = m.absorption.z;
        g.eta = m.eta;
        g.metallic = m.metallic;
        g.subsurface = m.subsurface;
        g.specular = m.specular;
        g.roughness = m.roughness;
        g.specularTint = m.specularTint;
        g.anisotropic = m.anisotropic;
        g.sheen = m.sheen;
        g.sheenTint = m.sheenTint;
        g.clearcoat = m.clearcoat;
        g.clearcoatGloss = m.clearcoatGloss;
        g.transmission = m.transmission;
        g.opacity = m.opacity;
        // Sentinel, not m.xxxTextureIndex - this helper feeds the isolated
        // BSDF unit test (BsdfTestMain.slang), which never samples textures;
        // -1 keeps it well-defined regardless of what `m` happens to carry.
        g.albedoTextureIndex = -1;
        g.roughnessTextureIndex = -1;
        g.metallicTextureIndex = -1;
        g.opacityTextureIndex = -1;
        return g;
    }

    bool RunBsdfValidation()
    {
        const int gridW = 64;
        const int gridH = 64;
        const int count = gridW * gridH;

        std::vector<BsdfTestInput> inputs;
        inputs.reserve(count);
        std::vector<GpuBsdfTestCase> gpuCases;
        gpuCases.reserve(count);

        for (int iy = 0; iy < gridH; ++iy)
        {
            for (int ix = 0; ix < gridW; ++ix)
            {
                BsdfTestInput in = GenerateBsdfTestCase(ix, iy, gridW, gridH);
                inputs.push_back(in);

                GpuBsdfTestCase gc{};
                gc.material = ToGpuMaterial(in.material);
                gc.etaI = in.etaI; gc.etaO = in.etaO;
                gc.nx = in.n.x; gc.ny = in.n.y; gc.nz = in.n.z;
                gc.vx = in.v.x; gc.vy = in.v.y; gc.vz = in.v.z;
                gc.lx = in.l.x; gc.ly = in.l.y; gc.lz = in.l.z;
                gc.seed = in.seed;
                gpuCases.push_back(gc);
            }
        }

        std::vector<GpuBsdfTestResult> gpuResults(count);
        {
            std::unique_ptr<Scene> dummyScene(new Scene());
            std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(dummyScene.get())));
            if (!gpuRenderer->IsAvailable())
            {
                fprintf(stderr, "FAIL: VulkanRenderer not available on this machine\n");
                return false;
            }
            gpuRenderer->RunBsdfTest(gpuCases.data(), gpuResults.data(), count);
        }

        double sumSqEval = 0.0, sumSqPdf = 0.0, sumSqSampleDir = 0.0, sumSqSamplePdf = 0.0;
        int badEval = 0, badPdf = 0, badSampleDir = 0, badSamplePdf = 0, badType = 0;
        const float kEvalTol = 1.e-3f;
        const float kPdfTol = 1.e-3f;
        const float kSampleDirTol = 1.e-3f;

        // pdf values are not well-conditioned as an *absolute* comparison: the
        // specular GGX pdf term divides by Dot(L,half), which can be small, so a
        // sub-ULP difference in the sampled direction (sin/cos evaluated by
        // different math libraries on CPU vs GPU) can amplify into a much larger
        // absolute pdf delta near grazing/degenerate configurations even though
        // the underlying direction and eval both match to ~1e-6. Compare pdfs
        // with a relative+absolute tolerance instead (numpy allclose-style).
        auto closeEnough = [](float a, float b, float atol, float rtol) {
            return fabsf(a - b) <= atol + rtol * fmaxf(fabsf(a), fabsf(b));
        };

        for (int i = 0; i < count; ++i)
        {
            const BsdfTestInput &in = inputs[i];
            const GpuBsdfTestResult &gr = gpuResults[i];

            Vec3 cpuEval = BSDFEval(in.material, in.etaI, in.etaO, Vec3(0.0f), in.n, in.v, in.l);
            float cpuPdf = BSDFPdf(in.material, in.etaI, in.etaO, Vec3(0.0f), in.n, in.v, in.l);

            Vec3 u, v;
            BasisFromVector(in.n, &u, &v);
            Random rand(in.seed);
            Vec3 cpuSampleDir;
            float cpuSamplePdf;
            BSDFType cpuType;
            BSDFSample(in.material, in.etaI, in.etaO, Vec3(0.0f), u, v, in.n, in.v, cpuSampleDir, cpuSamplePdf, cpuType, rand);

            float dEval = Length(cpuEval - Vec3(gr.evalX, gr.evalY, gr.evalZ));
            float dPdf = fabsf(cpuPdf - gr.pdf);
            float dSampleDir = Length(cpuSampleDir - Vec3(gr.sampleDirX, gr.sampleDirY, gr.sampleDirZ));
            float dSamplePdf = fabsf(cpuSamplePdf - gr.samplePdf);

            sumSqEval += dEval * dEval;
            sumSqPdf += dPdf * dPdf;
            sumSqSampleDir += dSampleDir * dSampleDir;
            sumSqSamplePdf += dSamplePdf * dSamplePdf;

            if (dEval > kEvalTol) ++badEval;
            if (dPdf > kPdfTol) ++badPdf;
            if (dSampleDir > kSampleDirTol) ++badSampleDir;
            if (!closeEnough(cpuSamplePdf, gr.samplePdf, 1.e-3f, 1.e-2f)) ++badSamplePdf;
            if ((int)cpuType != gr.sampleType) ++badType;
        }

        printf("[M2 BSDF] eval RMSE=%f pdf RMSE=%f sampleDir RMSE=%f samplePdf RMSE=%f\n",
               sqrt(sumSqEval / count), sqrt(sumSqPdf / count), sqrt(sumSqSampleDir / count), sqrt(sumSqSamplePdf / count));
        printf("[M2 BSDF] bad: eval=%d/%d pdf=%d/%d sampleDir=%d/%d samplePdf=%d/%d type-mismatch=%d/%d\n",
               badEval, count, badPdf, count, badSampleDir, count, badSamplePdf, count, badType, count);

        // Allow a small number of mismatches at BSDF branch boundaries (e.g. a
        // transmission-probability coin-flip landing on opposite sides of 0/1
        // due to float rounding at the extreme roughness/transmission grid
        // corners) without failing the whole run.
        const int kBadCountThreshold = count / 200; // ~0.5%
        bool pass = badEval <= kBadCountThreshold && badPdf <= kBadCountThreshold &&
                    badSampleDir <= kBadCountThreshold && badSamplePdf <= kBadCountThreshold &&
                    badType <= kBadCountThreshold;
        printf("[M2 BSDF] %s\n", pass ? "PASS" : "FAIL");
        return pass;
    }

    // Small synthetic HDR probe: a smooth gradient with no zero-total-weight
    // rows/columns (Probe::BuildCDF() divides by each row/column's total
    // weight, so an all-zero row would be a div-by-zero on the CPU side too).
    Probe MakeTestProbe(int width, int height)
    {
        Probe p;
        p.width = width;
        p.height = height;
        p.data = new Color[width * height];
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float u = float(x) / float(width);
                float v = float(y) / float(height);
                p.data[y * width + x] = Color(0.2f + 0.8f * u, 0.2f + 0.6f * v, 0.3f + 0.4f * (1.0f - u), 1.0f);
            }
        }
        p.BuildCDF();
        return p;
    }

    // M3 test scene: a diffuse sphere at the origin, a diffuse mesh quad beside
    // it, and an emissive sphere light - enough to exercise Trace's two-level
    // BVH, SampleLights' area-light NEE+MIS, and BSDFSample-driven indirect
    // bounces together. Materials are kept transmission=0/subsurface=0 (the
    // Material() defaults) so BSDFSample's branch decisions are deterministic
    // (their probabilities are exactly 0) rather than borderline float
    // comparisons against a coin-flip - see the plan's M3 notes on why that
    // keeps this a tight, not just statistical, comparison. Mesh/probe-based
    // lights and non-zero transmission/subsurface are reasonable follow-ups,
    // not required to validate the integrator's core control flow.
    // useProbe: when true, the sky uses an HDR environment probe (exercising
    // SampleLights' probe-NEE loop and PathTrace's probe-MIS miss-weighting
    // branch, both in kernels/slang/integrator/*.slang) instead of the analytic
    // horizon/zenith gradient - see RunPathTraceValidation()'s two calls.
    void BuildPathTraceTestScene(Scene &scene, bool useProbe)
    {
        Primitive spherePrim;
        spherePrim.type = eSphere;
        spherePrim.sphere.radius = 1.0f;
        spherePrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
        spherePrim.endTransform = spherePrim.startTransform;
        {
            auto sphereMat = std::make_unique<Material>();
            sphereMat->color = Vec3(0.7f, 0.3f, 0.3f);
            sphereMat->roughness = 0.4f;
            spherePrim.materialIndex = scene.AddMaterial(std::move(sphereMat));
        }
        scene.AddPrimitive(spherePrim);

        std::unique_ptr<Mesh> mesh = MakeQuadMesh();
        Primitive meshPrim;
        meshPrim.type = eMesh;
        meshPrim.startTransform = Transform(Vec3(2.3f, 0.0f, 0.0f));
        meshPrim.endTransform = meshPrim.startTransform;
        meshPrim.mesh = GeometryFromMesh(mesh.get());
        {
            auto meshMat = std::make_unique<Material>();
            meshMat->color = Vec3(0.3f, 0.5f, 0.7f);
            meshMat->roughness = 0.6f;
            meshPrim.materialIndex = scene.AddMaterial(std::move(meshMat));
        }
        meshPrim.mesh.meshIndex = scene.AddMesh(std::move(mesh));
        scene.AddPrimitive(meshPrim);

        Primitive lightPrim;
        lightPrim.type = eSphere;
        lightPrim.sphere.radius = 0.5f;
        lightPrim.startTransform = Transform(Vec3(0.0f, 3.0f, 1.0f));
        lightPrim.endTransform = lightPrim.startTransform;
        {
            auto lightMat = std::make_unique<Material>();
            lightMat->emission = Vec3(8.0f, 8.0f, 7.0f);
            lightPrim.materialIndex = scene.AddMaterial(std::move(lightMat));
        }
        lightPrim.lightSamples = 2;
        scene.AddPrimitive(lightPrim);

        if (useProbe)
        {
            scene.sky.probe = MakeTestProbe(24, 12);
        }
        else
        {
            scene.sky.horizon = Vec3(0.6f, 0.7f, 0.9f);
            scene.sky.zenith = Vec3(0.1f, 0.2f, 0.5f);
        }

        scene.Build();
    }

    // Tier-2 "punctual lights + rect/disk area lights": a diffuse sphere at
    // the origin (same as BuildPathTraceTestScene, so GeneratePathTraceTestCase's
    // existing ray fan hits real geometry), lit by one of each new light
    // type - a point light, a directional light, a rect area light, and a
    // disk area light - so this scene exercises every new CPU/GPU code path
    // added by that feature in a single RunPathTraceValidation() comparison:
    // Intersection.h/Primitive.slang's eRect/eDisk cases (via the rect/disk
    // Primitives being both directly visible and NEE-sampled), and
    // PunctualLight.h/PunctualLight.slang's no-MIS NEE block (via
    // scene.lights). Transmission/subsurface stay at Material() defaults for
    // the same "deterministic BSDF branch decisions" reason as
    // BuildPathTraceTestScene's comment explains.
    void BuildPunctualAndAreaLightTestScene(Scene &scene)
    {
        Primitive spherePrim;
        spherePrim.type = eSphere;
        spherePrim.sphere.radius = 1.0f;
        spherePrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
        spherePrim.endTransform = spherePrim.startTransform;
        {
            auto sphereMat = std::make_unique<Material>();
            sphereMat->color = Vec3(0.6f, 0.6f, 0.6f);
            sphereMat->roughness = 0.5f;
            spherePrim.materialIndex = scene.AddMaterial(std::move(sphereMat));
        }
        scene.AddPrimitive(spherePrim);

        // Rect area light, facing down toward the sphere (local +Z rotated
        // to -Y via a 90-degree rotation about X).
        {
            Primitive rectPrim;
            rectPrim.type = eRect;
            rectPrim.rect.width = 2.0f;
            rectPrim.rect.height = 2.0f;
            rectPrim.startTransform = Transform(Vec3(0.0f, 3.0f, 1.0f), Quat(0.7071068f, 0.0f, 0.0f, 0.7071068f));
            rectPrim.endTransform = rectPrim.startTransform;
            auto rectMat = std::make_unique<Material>();
            rectMat->emission = Vec3(4.0f, 4.0f, 3.5f);
            rectPrim.materialIndex = scene.AddMaterial(std::move(rectMat));
            rectPrim.lightSamples = 2;
            scene.AddPrimitive(rectPrim);
        }

        // Disk area light, off to one side.
        {
            Primitive diskPrim;
            diskPrim.type = eDisk;
            diskPrim.disk.radius = 0.6f;
            diskPrim.startTransform = Transform(Vec3(-2.0f, 1.5f, 2.0f), Quat(0.5f, 0.5f, 0.5f, 0.5f));
            diskPrim.endTransform = diskPrim.startTransform;
            auto diskMat = std::make_unique<Material>();
            diskMat->emission = Vec3(3.0f, 2.0f, 2.0f);
            diskPrim.materialIndex = scene.AddMaterial(std::move(diskMat));
            diskPrim.lightSamples = 2;
            scene.AddPrimitive(diskPrim);
        }

        // Point light.
        {
            PunctualLight point;
            point.type = PunctualLightType::ePoint;
            point.position = Vec3(2.5f, 2.0f, 2.5f);
            point.color = Vec3(1.0f, 0.5f, 0.5f);
            point.intensity = 20.0f;
            scene.lights.push_back(point);
        }

        // Directional light.
        {
            PunctualLight sun;
            sun.type = PunctualLightType::eDirectional;
            sun.direction = Normalize(Vec3(-0.3f, -1.0f, -0.4f));
            sun.color = Vec3(0.5f, 0.5f, 1.0f);
            sun.intensity = 1.2f;
            scene.lights.push_back(sun);
        }

        scene.sky.horizon = Vec3(0.05f, 0.05f, 0.07f);
        scene.sky.zenith = Vec3(0.01f, 0.01f, 0.02f);

        scene.Build();
    }

    struct PathTraceTestInput
    {
        Vec3 origin, dir;
        float time;
        int maxDepth;
        uint32_t seed;
    };

    // Explicit origin/dir per case (not derived via CameraSampler) - sidesteps
    // the pre-existing CameraSampler frustum quirk noted in M1 entirely, since
    // there is no camera involved in this comparison at all.
    PathTraceTestInput GeneratePathTraceTestCase(int ix, int iy, int gridW, int gridH)
    {
        PathTraceTestInput in{};
        in.origin = Vec3(0.0f, 0.0f, 6.0f);

        float yaw = DegToRad(-20.0f + 40.0f * (float(ix) / float(gridW - 1)));
        float pitch = DegToRad(-20.0f + 40.0f * (float(iy) / float(gridH - 1)));
        in.dir = Normalize(Vec3(sinf(yaw) * cosf(pitch), sinf(pitch), -cosf(yaw) * cosf(pitch)));

        in.time = 1.0f;
        in.maxDepth = 3;
        in.seed = (uint32_t)((ix * gridH + iy) * 2246822519u + 1u);
        return in;
    }

    // useProbe: see BuildPathTraceTestScene() - runs the same comparison twice
    // from main() (analytic sky, then HDR-probe sky) so both Sky_Eval() paths
    // and both SampleLights()/PathTrace() probe branches get exercised, not
    // just the isolated ProbeSample/Eval/Pdf functions RunProbeValidation()
    // already covers.
    // atol is caller-supplied (not derived here) since the right tolerance
    // depends on which known, pre-approved CPU/GPU deviations the scene
    // exercises (analytic-sky: none, tight; probe-sky: ProbeEval's bilinear-
    // vs-nearest deviation; textured-mesh: g_Textures[]'s hardware bilinear
    // quantization, see RunTextureSamplingValidation()'s M8 section) - same
    // reasoning as the two original call sites' differing atol, just made an
    // explicit parameter instead of an internal useProbe branch now that a
    // third, non-probe-related deviation source exists.
    bool RunPathTraceValidation(const std::function<void(Scene &)> &buildScene, float atol, const char *label)
    {
        const int gridW = 32;
        const int gridH = 32;
        const int count = gridW * gridH;

        Scene scene;
        buildScene(scene);
        uint32_t numPrimitives = (uint32_t)scene.primitives.size();
        // Tier-2 "punctual lights" - scene.lights isn't part of the BVH/
        // primitives array, so it needs its own explicit count, mirroring
        // numPrimitives above (see GpuFrameParams.numLights/
        // GpuPathTraceTestCase.numLights). The CPU side needs no equivalent
        // here - PathTrace(scene, ...) below reads scene.lights directly.
        uint32_t numLights = (uint32_t)scene.lights.size();

        std::vector<PathTraceTestInput> inputs;
        std::vector<GpuPathTraceTestCase> gpuCases;
        inputs.reserve(count);
        gpuCases.reserve(count);

        for (int iy = 0; iy < gridH; ++iy)
        {
            for (int ix = 0; ix < gridW; ++ix)
            {
                PathTraceTestInput in = GeneratePathTraceTestCase(ix, iy, gridW, gridH);
                inputs.push_back(in);

                GpuPathTraceTestCase gc{};
                gc.originX = in.origin.x; gc.originY = in.origin.y; gc.originZ = in.origin.z;
                gc.dirX = in.dir.x; gc.dirY = in.dir.y; gc.dirZ = in.dir.z;
                gc.time = in.time;
                gc.maxDepth = in.maxDepth;
                gc.seed = in.seed;
                gc.numPrimitives = numPrimitives;
                gc.numLights = numLights;
                gpuCases.push_back(gc);
            }
        }

        std::vector<GpuPathTraceTestResult> gpuResults(count);
        {
            std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
            if (!gpuRenderer->IsAvailable())
            {
                fprintf(stderr, "FAIL: VulkanRenderer not available on this machine\n");
                return false;
            }
            gpuRenderer->RunPathTraceTest(gpuCases.data(), gpuResults.data(), count);
        }

        // Radiance spans a wide dynamic range (0 for misses, ~8-10 for direct
        // light hits) - relative+absolute tolerance again, same reasoning as
        // M2's samplePdf comparison.
        auto closeEnough = [](float a, float b, float atol, float rtol) {
            return fabsf(a - b) <= atol + rtol * fmaxf(fabsf(a), fabsf(b));
        };

        double sumSq = 0.0;
        float maxErr = 0.0f;
        int bad = 0;

        for (int i = 0; i < count; ++i)
        {
            const PathTraceTestInput &in = inputs[i];
            const GpuPathTraceTestResult &gr = gpuResults[i];

            Random rand(in.seed);
            Vec3 cpuRadiance = PathTrace(scene, in.origin, in.dir, in.time, in.maxDepth, rand);
            Vec3 gpuRadiance(gr.radianceX, gr.radianceY, gr.radianceZ);

            float d = Length(cpuRadiance - gpuRadiance);
            sumSq += d * d;
            maxErr = Max(maxErr, d);

            bool ok = closeEnough(cpuRadiance.x, gpuRadiance.x, atol, 2.e-2f) &&
                      closeEnough(cpuRadiance.y, gpuRadiance.y, atol, 2.e-2f) &&
                      closeEnough(cpuRadiance.z, gpuRadiance.z, atol, 2.e-2f);
            if (!ok)
                ++bad;
        }

        printf("[M3 PathTrace %s] RMSE=%f maxErr=%f bad=%d/%d\n", label, sqrt(sumSq / count), maxErr, bad, count);

        const int kBadCountThreshold = count / 100; // ~1%
        bool pass = bad <= kBadCountThreshold;
        printf("[M3 PathTrace %s] %s\n", label, pass ? "PASS" : "FAIL");
        return pass;
    }

    // Non-gating sanity check for the real per-pixel RenderMode::ePathTrace
    // dispatch (PathTraceMain.slang via VulkanRenderer::Render()) - the piece
    // RunPathTraceValidation() above doesn't exercise (it calls PathTrace()
    // directly, bypassing CameraSampler ray generation and AddSample's
    // progressive accumulation/resolve). Per-pixel noise is expected to differ
    // from the CPU renderer (independent RNG streams - see PathTraceMain.slang's
    // header comment), so this only checks "runs without error and produces
    // plausible, non-degenerate output", not a CPU-vs-GPU pixel match.
    void RunPathTraceSmokeTest()
    {
        const int width = 128, height = 128;

        Scene scene;
        BuildPathTraceTestScene(scene, false);

        Camera camera;
        camera.position = Vec3(0.0f, 0.5f, 6.0f);
        camera.fov = DegToRad(45.0f);

        Options options{};
        options.mode = ePathTrace;
        options.width = width;
        options.height = height;
        options.enableDOF = false;
        options.maxDepth = 3;
        options.maxSamples = 1;
        options.exposure = 1.0f;
        options.limit = 1000.0f;
        options.clamp = 20.0f;

        std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
        if (!gpuRenderer->IsAvailable())
        {
            fprintf(stderr, "[M3 PathTrace smoke test] SKIP: VulkanRenderer not available\n");
            return;
        }
        gpuRenderer->Init(width, height);

        std::vector<Color> image(width * height, Color(0.0f));
        const int samples = 8;
        for (int s = 0; s < samples; ++s)
            gpuRenderer->Render(camera, options, image.data());

        double sum = 0.0;
        int nonBlack = 0;
        for (auto &c : image)
        {
            sum += c.x + c.y + c.z;
            if (c.x > 0.001f || c.y > 0.001f || c.z > 0.001f)
                ++nonBlack;
        }

        WritePng("kernel_validate_pathtrace.png", width, height, image);
        printf("[M3 PathTrace smoke test] %d samples accumulated, avg luminance sum=%f, non-black pixels=%d/%d "
               "(image written to kernel_validate_pathtrace.png for visual inspection)\n",
               samples, sum / (width * height), nonBlack, width * height);
    }

    // M9 (Tier-1 "deforming/skinned-mesh motion blur"): builds a single
    // deforming quad mesh - vertices/verticesEnd differ by a known local-space
    // translation, so the quad's world-space center sweeps from x=0 (time=0)
    // to x=kDeformDeltaX (time=1) - then fires an explicit ray at a fixed
    // world-space X coordinate the swept quad only crosses for a sub-range of
    // the shutter. Unlike RunPathTraceValidation's grid generator (which
    // always uses time=1.0, see GeneratePathTraceTestCase), this varies
    // `time` directly, so it's the one test in this file that actually
    // exercises the mid-shutter vertex lerp itself (Intersection.h's
    // MeshQuery / geometry/Primitive.slang's IntersectRayMesh), not just the
    // verticesEnd endpoint. Comparing against an analytically-known hit/miss
    // pattern (not just CPU-vs-GPU mutual agreement) also catches a bug that
    // happens to be reproduced identically on both backends.
    namespace
    {
        const float kDeformDeltaX = 3.0f;
        const float kQuadHalfWidth = 0.7f; // MakeQuadMesh()'s own vertex extent
        const float kRayX = 1.5f;          // crosses the swept quad only for t in ~[0.27, 0.73]

        std::unique_ptr<Mesh> MakeDeformingQuadMesh()
        {
            std::unique_ptr<Mesh> mesh = MakeQuadMesh();
            mesh->verticesEnd = mesh->vertices;
            for (auto &v : mesh->verticesEnd)
                v.x += kDeformDeltaX;
            mesh->rebuildBVH(); // rebuild with the swept per-triangle bounds now that verticesEnd is set
            return mesh;
        }

        void BuildDeformingMeshTestScene(Scene &scene)
        {
            std::unique_ptr<Mesh> mesh = MakeDeformingQuadMesh();

            Primitive meshPrim;
            meshPrim.type = eMesh;
            meshPrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
            meshPrim.endTransform = meshPrim.startTransform;
            meshPrim.mesh = GeometryFromMesh(mesh.get());
            {
                // Strong, distinctive emission and no other light/sky in the
                // scene, so "hit the quad" vs "hit nothing" is an unambiguous
                // radiance threshold, not a subtle color comparison.
                auto mat = std::make_unique<Material>();
                mat->color = Vec3(0.0f);
                mat->emission = Vec3(5.0f, 1.0f, 1.0f);
                meshPrim.materialIndex = scene.AddMaterial(std::move(mat));
            }
            meshPrim.mesh.meshIndex = scene.AddMesh(std::move(mesh));
            scene.AddPrimitive(meshPrim);

            scene.Build();
        }

        // Analytic ground truth: quad center at time t is (kDeformDeltaX * t,
        // 0, 0) with half-width kQuadHalfWidth (primitive transform is
        // identity, so local and world space coincide).
        bool ExpectHitAtTime(float t)
        {
            float centerX = kDeformDeltaX * t;
            return fabsf(kRayX - centerX) <= kQuadHalfWidth;
        }
    }

    bool RunMotionBlurValidation()
    {
        Scene scene;
        BuildDeformingMeshTestScene(scene);
        uint32_t numPrimitives = (uint32_t)scene.primitives.size();

        const int kNumTimeSamples = 21; // t = 0.00, 0.05, ..., 1.00
        const Vec3 origin(kRayX, 0.0f, 6.0f);
        const Vec3 dir(0.0f, 0.0f, -1.0f);

        std::vector<GpuPathTraceTestCase> gpuCases;
        gpuCases.reserve(kNumTimeSamples);
        for (int i = 0; i < kNumTimeSamples; ++i)
        {
            float t = float(i) / float(kNumTimeSamples - 1);

            GpuPathTraceTestCase gc{};
            gc.originX = origin.x; gc.originY = origin.y; gc.originZ = origin.z;
            gc.dirX = dir.x; gc.dirY = dir.y; gc.dirZ = dir.z;
            gc.time = t;
            gc.maxDepth = 1;
            gc.seed = (uint32_t)(i * 2246822519u + 1u);
            gc.numPrimitives = numPrimitives;
            gpuCases.push_back(gc);
        }

        std::vector<GpuPathTraceTestResult> gpuResults(kNumTimeSamples);
        {
            std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
            if (!gpuRenderer->IsAvailable())
            {
                fprintf(stderr, "[M9 Motion blur] FAIL: VulkanRenderer not available\n");
                return false;
            }
            gpuRenderer->RunPathTraceTest(gpuCases.data(), gpuResults.data(), kNumTimeSamples);
        }

        bool pass = true;
        for (int i = 0; i < kNumTimeSamples; ++i)
        {
            float t = float(i) / float(kNumTimeSamples - 1);
            const GpuPathTraceTestCase &gc = gpuCases[i];
            const GpuPathTraceTestResult &gr = gpuResults[i];

            Random rand(gc.seed);
            Vec3 cpuRadiance = PathTrace(scene, origin, dir, t, gc.maxDepth, rand);
            Vec3 gpuRadiance(gr.radianceX, gr.radianceY, gr.radianceZ);

            bool expectHit = ExpectHitAtTime(t);
            bool cpuHit = cpuRadiance.x > 2.5f; // emission.x=5.0 on hit, 0 on miss/background
            bool gpuHit = gpuRadiance.x > 2.5f;
            bool agree = Length(cpuRadiance - gpuRadiance) < 0.05f;

            if (cpuHit != expectHit || gpuHit != expectHit || !agree)
            {
                pass = false;
                printf("[M9 Motion blur] FAIL t=%.3f expectHit=%d cpuHit=%d gpuHit=%d "
                       "cpu=(%.3f,%.3f,%.3f) gpu=(%.3f,%.3f,%.3f)\n",
                       t, expectHit, cpuHit, gpuHit,
                       cpuRadiance.x, cpuRadiance.y, cpuRadiance.z,
                       gpuRadiance.x, gpuRadiance.y, gpuRadiance.z);
            }
        }

        printf("[M9 Motion blur] deforming-mesh %s\n", pass ? "PASS" : "FAIL");

        // Visual sanity check for a human reviewer, supplementary to the
        // analytic pass/fail above: two CPU-only renders of the same scene
        // with the shutter pinned to a single instant each (shutterStart ==
        // shutterEnd, so every accumulated sample sees the same time - no
        // inter-sample blending), showing the quad at its start-of-shutter
        // and end-of-shutter positions.
        {
            const int width = 160, height = 90;
            Camera camera;
            camera.position = Vec3(kDeformDeltaX * 0.5f, 0.0f, 8.0f);
            camera.fov = DegToRad(55.0f);

            Options options{};
            options.mode = ePathTrace;
            options.width = width;
            options.height = height;
            options.maxDepth = 1;
            options.maxSamples = 1;
            options.exposure = 1.0f;
            options.clamp = 20.0f;

            std::unique_ptr<Renderer> cpuRenderer(CreateCpuRenderer(&scene));
            for (float t : {0.0f, 1.0f})
            {
                camera.shutterStart = t;
                camera.shutterEnd = t;
                std::vector<Color> image(width * height, Color(0.0f));
                cpuRenderer->Render(camera, options, image.data());
                char path[64];
                snprintf(path, sizeof(path), "kernel_validate_motion_blur_t%d.png", (int)t);
                WritePng(path, width, height, image);
            }
            printf("[M9 Motion blur] wrote kernel_validate_motion_blur_t0.png / _t1.png "
                   "(quad should visibly shift right between the two)\n");
        }

        return pass;
    }

    // M10 (Tier-1 "instancing/particles"): builds one scene combining both
    // PointInstancer use cases - a shared mesh instanced several times (the
    // "instancing" case) and a small grid of emissive spheres (the
    // "particles" case) - then checks three things: (1) Scene::Build()
    // expanded the instancers into exactly the right number of Primitives,
    // (2) every expanded mesh Primitive really does share one Mesh/meshIndex
    // rather than each getting its own copy (the one thing ExpandInstancers()
    // does that no other code path in this file exercises - every other mesh
    // test scene is 1:1 mesh<->primitive), and (3) the expanded scene renders
    // identically on CPU and GPU, reusing RunPathTraceValidation exactly like
    // every other scene-level check in this file - instancing adds no new
    // traversal/intersection code, so this is really validating "expansion
    // produced correct, ordinary Primitives", not a new rendering path.
    namespace
    {
        const int kInstancedQuadCount = 5;
        const int kInstancedSphereCountX = 3;
        const int kInstancedSphereCountY = 2;

        void BuildInstancedTestScene(Scene &scene)
        {
            // Instancing case: one shared quad mesh, stamped out
            // kInstancedQuadCount times along X.
            std::unique_ptr<Mesh> mesh = MakeQuadMesh();
            int meshIndex = scene.AddMesh(std::move(mesh));

            PointInstancer quadInstancer;
            quadInstancer.type = eMesh;
            quadInstancer.meshIndex = meshIndex;
            {
                auto mat = std::make_unique<Material>();
                mat->color = Vec3(0.3f, 0.5f, 0.7f);
                mat->roughness = 0.5f;
                quadInstancer.materialIndex = scene.AddMaterial(std::move(mat));
            }
            for (int i = 0; i < kInstancedQuadCount; ++i)
            {
                float x = -2.0f + float(i); // -2, -1, 0, 1, 2
                quadInstancer.instanceStart.push_back(Transform(Vec3(x, 0.0f, 0.0f), Quat(), 0.35f));
            }
            // instanceEnd left empty - static instances (no per-instance
            // motion blur exercised here; that's the same Transform
            // interpolation every other Primitive already uses, not
            // instancer-specific behavior worth re-testing).
            scene.AddInstancer(quadInstancer);

            // Particles case: a small grid of small emissive spheres.
            PointInstancer sphereInstancer;
            sphereInstancer.type = eSphere;
            sphereInstancer.sphereRadius = 0.2f;
            {
                auto mat = std::make_unique<Material>();
                mat->color = Vec3(0.0f);
                mat->emission = Vec3(2.0f, 1.5f, 0.5f);
                sphereInstancer.materialIndex = scene.AddMaterial(std::move(mat));
            }
            sphereInstancer.lightSamples = 1;
            for (int gx = 0; gx < kInstancedSphereCountX; ++gx)
                for (int gy = 0; gy < kInstancedSphereCountY; ++gy)
                    sphereInstancer.instanceStart.push_back(
                        Transform(Vec3(-0.6f + 0.6f * gx, 1.5f + 0.6f * gy, 0.5f)));
            scene.AddInstancer(sphereInstancer);

            scene.sky.horizon = Vec3(0.3f, 0.35f, 0.45f);
            scene.sky.zenith = Vec3(0.05f, 0.1f, 0.25f);

            scene.Build();
        }
    }

    bool RunInstancingValidation()
    {
        bool pass = true;

        Scene scene;
        BuildInstancedTestScene(scene);

        const size_t expectedPrimitives = kInstancedQuadCount + (size_t)kInstancedSphereCountX * kInstancedSphereCountY;
        if (scene.primitives.size() != expectedPrimitives)
        {
            pass = false;
            printf("[M10 Instancing] FAIL: expected %zu expanded primitives, got %zu\n",
                   expectedPrimitives, scene.primitives.size());
        }

        // instancers must be consumed (cleared) by Build() - re-Build()'ing
        // must not re-expand and duplicate the same instances.
        if (!scene.instancers.empty())
        {
            pass = false;
            printf("[M10 Instancing] FAIL: Scene::instancers not cleared after Build()\n");
        }

        const Mesh *sharedMesh = nullptr;
        int meshPrimCount = 0;
        for (const Primitive &p : scene.primitives)
        {
            if (p.type != eMesh)
                continue;
            ++meshPrimCount;
            if (p.mesh.meshIndex < 0)
            {
                pass = false;
                printf("[M10 Instancing] FAIL: expanded mesh primitive has unset meshIndex\n");
            }
            const Mesh *thisMesh = (const Mesh *)p.mesh.id;
            if (!sharedMesh)
                sharedMesh = thisMesh;
            else if (thisMesh != sharedMesh)
            {
                pass = false;
                printf("[M10 Instancing] FAIL: expanded mesh primitives don't all share one Mesh\n");
            }
        }
        if (meshPrimCount != kInstancedQuadCount)
        {
            pass = false;
            printf("[M10 Instancing] FAIL: expected %d mesh primitives, got %d\n", kInstancedQuadCount, meshPrimCount);
        }

        printf("[M10 Instancing] expansion checks %s (%zu primitives, %d sharing one mesh)\n",
               pass ? "PASS" : "FAIL", scene.primitives.size(), meshPrimCount);

        bool renderPass = RunPathTraceValidation(
            [](Scene &s) { BuildInstancedTestScene(s); }, 1.e-2f, "instanced");
        pass = pass && renderPass;

        printf("[M10 Instancing] %s\n", pass ? "PASS" : "FAIL");

        // Visual sanity check for a human reviewer: a fresh scene (Build()
        // consumed the first one's instancers) rendered CPU-only, wide
        // enough to show all 5 quad instances and the sphere-particle grid
        // at once.
        {
            Scene visScene;
            BuildInstancedTestScene(visScene);

            const int width = 240, height = 135;
            Camera camera;
            camera.position = Vec3(0.0f, 0.7f, 6.0f);
            camera.fov = DegToRad(55.0f);

            Options options{};
            options.mode = ePathTrace;
            options.width = width;
            options.height = height;
            options.maxDepth = 2;
            options.maxSamples = 16;
            options.exposure = 1.0f;
            options.clamp = 20.0f;
            options.filter = Filter(FilterType::eFilterGaussian, 1.0f, 2.0f);

            std::unique_ptr<Renderer> cpuRenderer(CreateCpuRenderer(&visScene));
            std::vector<Color> image(width * height, Color(0.0f));
            cpuRenderer->Render(camera, options, image.data());
            WritePng("kernel_validate_instancing.png", width, height, image);
            printf("[M10 Instancing] wrote kernel_validate_instancing.png "
                   "(5 quad instances sharing one mesh + a 3x2 emissive sphere-particle grid)\n");
        }

        return pass;
    }

    // M11 (Tier-1 "opacity/cutout transparency"): two sub-tests.
    //
    // Sub-test 1 fires many independent-seed rays straight at a single
    // opacity=0.5 plane in front of a distinctly-colored sky - a plane
    // (not a sphere) deliberately, since a ray through a sphere crosses it
    // twice (TraceWithCutout would re-roll opacity at the far side too, per
    // IntersectRaySphere's "ray origin ended up inside the sphere" branch
    // in Intersection.h) which would turn the analytic expectation below
    // into (1-opacity)^2 instead of the clean 1-opacity this test wants.
    // Checks CPU/GPU agreement per-sample (same seed + algorithmically
    // identical RNG on both backends, per Random.slang's own header
    // comment, so this holds to the same near-bit-exact bar as every other
    // PathTrace validation case in this file) AND that the measured
    // hit-fraction/mean-radiance actually converges toward the configured
    // opacity - proof cutout is firing at roughly the right rate, not just
    // that both backends agree on being wrong the same way.
    //
    // Sub-test 2 exercises the *shadow*-ray cutout path specifically
    // (SampleLights' NEE shadow rays, not primary/camera rays) - a floor
    // lit by a light behind a cutout "screen", compared across
    // screenOpacity = 0.0 (no occluder) / 1.0 (fully blocked) / 0.5
    // (cutout), checking the fully-blocked case is exactly zero, the
    // cutout case is strictly between the two baselines (bleed-through is
    // happening, not fully blocked and not fully unoccluded), and CPU/GPU
    // agree per-sample at each opacity.
    namespace
    {
        const float kOpacityTestValue = 0.5f;
        const int kOpacitySampleCount = 64;

        // A ray from (0,0,6) along -Z crosses this plane exactly once at
        // z=0 - see the single-vs-double-crossing note above.
        void BuildOpacityCutoutTestScene(Scene &scene)
        {
            Primitive cutoutPrim;
            cutoutPrim.type = ePlane;
            cutoutPrim.plane.plane[0] = 0.0f;
            cutoutPrim.plane.plane[1] = 0.0f;
            cutoutPrim.plane.plane[2] = 1.0f;
            cutoutPrim.plane.plane[3] = 0.0f; // z = 0
            cutoutPrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
            cutoutPrim.endTransform = cutoutPrim.startTransform;
            {
                auto mat = std::make_unique<Material>();
                mat->color = Vec3(0.0f);
                mat->emission = Vec3(3.0f, 0.2f, 0.2f); // strong, distinct red - "opaque hit" signal
                mat->opacity = kOpacityTestValue;
                cutoutPrim.materialIndex = scene.AddMaterial(std::move(mat));
            }
            scene.AddPrimitive(cutoutPrim);

            // Strong, distinct blue, constant regardless of direction
            // (horizon == zenith) - "saw through" signal.
            scene.sky.horizon = Vec3(0.2f, 0.2f, 3.0f);
            scene.sky.zenith = scene.sky.horizon;

            scene.Build();
        }

        // Floor (y=-1) lit by a small light (0,3,0) straight up from the
        // test ray's floor-hit point, through a "screen" plane at y=1 -
        // the vertical shadow ray crosses the screen exactly once, same
        // single-crossing reasoning as the plane above.
        void BuildShadowCutoutTestScene(Scene &scene, float screenOpacity)
        {
            Primitive floorPrim;
            floorPrim.type = ePlane;
            floorPrim.plane.plane[0] = 0.0f;
            floorPrim.plane.plane[1] = 1.0f;
            floorPrim.plane.plane[2] = 0.0f;
            floorPrim.plane.plane[3] = 1.0f; // y = -1
            floorPrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
            floorPrim.endTransform = floorPrim.startTransform;
            {
                auto mat = std::make_unique<Material>();
                mat->color = Vec3(0.6f);
                mat->roughness = 1.0f;
                floorPrim.materialIndex = scene.AddMaterial(std::move(mat));
            }
            scene.AddPrimitive(floorPrim);

            Primitive screenPrim;
            screenPrim.type = ePlane;
            screenPrim.plane.plane[0] = 0.0f;
            screenPrim.plane.plane[1] = 1.0f;
            screenPrim.plane.plane[2] = 0.0f;
            screenPrim.plane.plane[3] = -1.0f; // y = 1
            screenPrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
            screenPrim.endTransform = screenPrim.startTransform;
            {
                auto mat = std::make_unique<Material>();
                mat->color = Vec3(0.0f);
                mat->opacity = screenOpacity;
                screenPrim.materialIndex = scene.AddMaterial(std::move(mat));
            }
            scene.AddPrimitive(screenPrim);

            Primitive lightPrim;
            lightPrim.type = eSphere;
            lightPrim.sphere.radius = 0.4f;
            lightPrim.startTransform = Transform(Vec3(0.0f, 3.0f, 0.0f));
            lightPrim.endTransform = lightPrim.startTransform;
            {
                auto mat = std::make_unique<Material>();
                mat->emission = Vec3(8.0f);
                lightPrim.materialIndex = scene.AddMaterial(std::move(mat));
            }
            lightPrim.lightSamples = 4;
            scene.AddPrimitive(lightPrim);

            scene.Build();
        }
    }

    bool RunOpacityValidation()
    {
        bool pass = true;

        // --- Sub-test 1: primary-ray cutout convergence --------------------
        {
            Scene scene;
            BuildOpacityCutoutTestScene(scene);
            uint32_t numPrimitives = (uint32_t)scene.primitives.size();

            const Vec3 origin(0.0f, 0.0f, 6.0f);
            const Vec3 dir(0.0f, 0.0f, -1.0f);

            std::vector<GpuPathTraceTestCase> gpuCases;
            gpuCases.reserve(kOpacitySampleCount);
            for (int i = 0; i < kOpacitySampleCount; ++i)
            {
                GpuPathTraceTestCase gc{};
                gc.originX = origin.x; gc.originY = origin.y; gc.originZ = origin.z;
                gc.dirX = dir.x; gc.dirY = dir.y; gc.dirZ = dir.z;
                gc.time = 0.0f;
                gc.maxDepth = 1;
                gc.seed = (uint32_t)(i * 2246822519u + 1u);
                gc.numPrimitives = numPrimitives;
                gpuCases.push_back(gc);
            }

            std::vector<GpuPathTraceTestResult> gpuResults(kOpacitySampleCount);
            {
                std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
                if (!gpuRenderer->IsAvailable())
                {
                    fprintf(stderr, "[M11 Opacity] FAIL: VulkanRenderer not available\n");
                    return false;
                }
                gpuRenderer->RunPathTraceTest(gpuCases.data(), gpuResults.data(), kOpacitySampleCount);
            }

            int cpuHits = 0, gpuHits = 0;
            double sumSq = 0.0;
            int bad = 0;
            for (int i = 0; i < kOpacitySampleCount; ++i)
            {
                Random rand(gpuCases[i].seed);
                Vec3 cpuRadiance = PathTrace(scene, origin, dir, 0.0f, 1, rand);
                Vec3 gpuRadiance(gpuResults[i].radianceX, gpuResults[i].radianceY, gpuResults[i].radianceZ);

                bool cpuHit = cpuRadiance.x > 1.0f; // red channel: 3.0 on hit, 0.2 on see-through
                bool gpuHit = gpuRadiance.x > 1.0f;
                if (cpuHit) ++cpuHits;
                if (gpuHit) ++gpuHits;

                float d = Length(cpuRadiance - gpuRadiance);
                sumSq += (double)d * d;
                if (d > 1.e-3f || cpuHit != gpuHit)
                {
                    ++bad;
                    printf("[M11 Opacity] MISMATCH sample=%d cpu=(%.3f,%.3f,%.3f) gpu=(%.3f,%.3f,%.3f)\n",
                           i, cpuRadiance.x, cpuRadiance.y, cpuRadiance.z, gpuRadiance.x, gpuRadiance.y, gpuRadiance.z);
                }
            }

            printf("[M11 Opacity] primary-ray RMSE=%f bad=%d/%d\n", sqrt(sumSq / kOpacitySampleCount), bad, kOpacitySampleCount);
            pass = pass && (bad == 0);

            // Statistical convergence - standard error of a
            // Bernoulli(opacity) mean over kOpacitySampleCount trials is
            // sqrt(p(1-p)/N); 4 sigma is a generous, still-tight bound.
            float cpuHitFraction = float(cpuHits) / float(kOpacitySampleCount);
            float gpuHitFraction = float(gpuHits) / float(kOpacitySampleCount);
            float stdErr = sqrtf(kOpacityTestValue * (1.0f - kOpacityTestValue) / float(kOpacitySampleCount));
            bool convergeOk = fabsf(cpuHitFraction - kOpacityTestValue) < 4.0f * stdErr &&
                               fabsf(gpuHitFraction - kOpacityTestValue) < 4.0f * stdErr;
            printf("[M11 Opacity] hit fraction: expected~%.3f cpu=%.3f gpu=%.3f (stdErr=%.3f) %s\n",
                   kOpacityTestValue, cpuHitFraction, gpuHitFraction, stdErr, convergeOk ? "OK" : "MISMATCH");
            pass = pass && convergeOk;
        }

        // --- Sub-test 2: shadow bleed-through monotonicity -----------------
        {
            const float opacities[3] = {0.0f, 1.0f, kOpacityTestValue}; // no-occluder, fully-blocked, cutout
            Vec3 cpuMeans[3];
            const int kShadowSampleCount = 32;
            // Between the floor (y=-1) and the screen (y=1), not above the
            // screen - the primary ray must reach the floor unobstructed so
            // only the shadow ray (floor -> light) crosses the screen. An
            // origin above the screen would make the *primary* ray hit the
            // screen instead of the floor, invalidating this sub-test.
            const Vec3 origin(0.0f, 0.0f, 0.0f);
            const Vec3 dir(0.0f, -1.0f, 0.0f);

            for (int variant = 0; variant < 3; ++variant)
            {
                Scene shadowScene;
                BuildShadowCutoutTestScene(shadowScene, opacities[variant]);
                uint32_t numPrims = (uint32_t)shadowScene.primitives.size();

                std::vector<GpuPathTraceTestCase> gpuCases;
                gpuCases.reserve(kShadowSampleCount);
                for (int i = 0; i < kShadowSampleCount; ++i)
                {
                    GpuPathTraceTestCase gc{};
                    gc.originX = origin.x; gc.originY = origin.y; gc.originZ = origin.z;
                    gc.dirX = dir.x; gc.dirY = dir.y; gc.dirZ = dir.z;
                    gc.time = 0.0f;
                    gc.maxDepth = 1;
                    gc.seed = (uint32_t)((variant * 97 + i) * 2246822519u + 1u);
                    gc.numPrimitives = numPrims;
                    gpuCases.push_back(gc);
                }

                std::vector<GpuPathTraceTestResult> gpuResults(kShadowSampleCount);
                {
                    std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&shadowScene)));
                    if (!gpuRenderer->IsAvailable())
                    {
                        fprintf(stderr, "[M11 Opacity] FAIL: VulkanRenderer not available (shadow sub-test)\n");
                        return false;
                    }
                    gpuRenderer->RunPathTraceTest(gpuCases.data(), gpuResults.data(), kShadowSampleCount);
                }

                Vec3 cpuSum(0.0f);
                double sumSq = 0.0;
                int bad = 0;
                for (int i = 0; i < kShadowSampleCount; ++i)
                {
                    Random rand(gpuCases[i].seed);
                    Vec3 cpuRadiance = PathTrace(shadowScene, origin, dir, 0.0f, 1, rand);
                    Vec3 gpuRadiance(gpuResults[i].radianceX, gpuResults[i].radianceY, gpuResults[i].radianceZ);
                    cpuSum += cpuRadiance;

                    float d = Length(cpuRadiance - gpuRadiance);
                    sumSq += (double)d * d;
                    if (d > 1.e-3f)
                        ++bad;
                }
                cpuMeans[variant] = cpuSum / float(kShadowSampleCount);

                printf("[M11 Opacity] shadow screenOpacity=%.2f mean=(%.4f,%.4f,%.4f) RMSE=%f bad=%d/%d\n",
                       opacities[variant], cpuMeans[variant].x, cpuMeans[variant].y, cpuMeans[variant].z,
                       sqrt(sumSq / kShadowSampleCount), bad, kShadowSampleCount);
                pass = pass && (bad == 0);
            }

            float noOccluder = cpuMeans[0].x;
            float fullyBlocked = cpuMeans[1].x;
            float cutout = cpuMeans[2].x;

            bool blockedIsZero = fullyBlocked < 1.e-4f;
            bool bracketed = cutout > fullyBlocked + 1.e-3f && cutout < noOccluder - 1.e-3f;
            bool monotonicOk = blockedIsZero && bracketed;

            printf("[M11 Opacity] shadow bleed-through: no-occluder=%.4f cutout(0.5)=%.4f fully-blocked=%.4f %s\n",
                   noOccluder, cutout, fullyBlocked, monotonicOk ? "OK (strictly bracketed)" : "MISMATCH");
            pass = pass && monotonicOk;
        }

        printf("[M11 Opacity] %s\n", pass ? "PASS" : "FAIL");
        return pass;
    }

    // M12 (Tier-1 "bump/normal mapping"): three checks.
    //
    // (1) Hand-verified CPU-only math check of ApplyNormalMap itself, no
    // scene/texture/BVH involved - a flat (tangent-space "up") texel must
    // return the normal unchanged; a texel encoding a known 3-4-5-triangle
    // tilt, evaluated against a normal/tangent basis that exactly aligns
    // with world axes, has an exactly hand-computable expected result; a
    // texel that would flip the normal past 90 degrees from the
    // unperturbed one must trigger the safety-net fallback.
    //
    // (2) Regression: a flat normal map must render float-identical to no
    // normal map at all - a cheap, strong check that the decode+TBN
    // round-trip's identity case really is a no-op, not just "close".
    //
    // (3) CPU/GPU parity on a real (non-flat) normal map, reusing the
    // standard RunPathTraceValidation harness like every other scene-level
    // check in this file - exercises the full pipeline (tangent
    // computation/interpolation, Material_SampleNormal, ApplyNormalMap),
    // not just the isolated math from (1).
    namespace
    {
        bool CheckNormalMapping(const char *name, Vec3 got, Vec3 expected, float tol, bool &pass)
        {
            float d = Length(got - expected);
            bool ok = d < tol;
            printf("[M12 NormalMapping] %s: expected (%.4f,%.4f,%.4f) got (%.4f,%.4f,%.4f) %s\n",
                   name, expected.x, expected.y, expected.z, got.x, got.y, got.z, ok ? "OK" : "MISMATCH");
            pass = pass && ok;
            return ok;
        }

        bool RunNormalMappingMathValidation()
        {
            bool pass = true;

            Vec3 normal(0.0f, 0.0f, 1.0f);
            Vec3 tangent(1.0f, 0.0f, 0.0f);

            // Flat/resting texel (0.5,0.5,1.0) -> decodes to (0,0,1) -> no
            // perturbation.
            CheckNormalMapping("flat texel", ApplyNormalMap(normal, tangent, Vec3(0.5f, 0.5f, 1.0f)), normal, 1.e-5f, pass);

            // Texel (0.8,0.5,0.9) decodes to tangent-space (0.6,0,0.8) (a
            // unit 3-4-5-triangle vector) - with tangent=+X, normal=+Z
            // (bitangent = cross(normal,tangent) = +Y), the perturbed
            // world-space normal is exactly (0.6,0,0.8).
            CheckNormalMapping("tilted texel", ApplyNormalMap(normal, tangent, Vec3(0.8f, 0.5f, 0.9f)), Vec3(0.6f, 0.0f, 0.8f), 1.e-5f, pass);

            // Texel (1.0,0.5,0.0) decodes to tangent-space (1,0,-1) - the
            // resulting perturbed normal would face away from the
            // unperturbed one (Dot < 0); the safety net must fall back to
            // `normal` unchanged rather than returning a backward-facing
            // normal.
            CheckNormalMapping("flip safety net", ApplyNormalMap(normal, tangent, Vec3(1.0f, 0.5f, 0.0f)), normal, 1.e-5f, pass);

            printf("[M12 NormalMapping] math %s\n", pass ? "PASS" : "FAIL");
            return pass;
        }

        Texture MakeFlatNormalMapTexture()
        {
            Texture tex;
            tex.width = 2;
            tex.height = 2;
            tex.depth = 1;
            tex.data = new float[2 * 2 * 4];
            for (int i = 0; i < 4; ++i)
            {
                tex.data[i * 4 + 0] = 0.5f;
                tex.data[i * 4 + 1] = 0.5f;
                tex.data[i * 4 + 2] = 1.0f;
                tex.data[i * 4 + 3] = 1.0f;
            }
            return tex;
        }

        // Constant tangent-space tilt toward +tangent, decoded value
        // (0.6,0.0,0.8) - same 3-4-5-triangle encoding as the math check
        // above, chosen so a raking light visibly changes the shading of a
        // normal-mapped quad relative to the flat case.
        Texture MakeTiltedNormalMapTexture()
        {
            Texture tex;
            tex.width = 2;
            tex.height = 2;
            tex.depth = 1;
            tex.data = new float[2 * 2 * 4];
            for (int i = 0; i < 4; ++i)
            {
                tex.data[i * 4 + 0] = 0.8f; // decodes to 0.6
                tex.data[i * 4 + 1] = 0.5f; // decodes to 0.0
                tex.data[i * 4 + 2] = 0.9f; // decodes to 0.8
                tex.data[i * 4 + 3] = 1.0f;
            }
            return tex;
        }

        // Same quad shape/UVs as BuildTexturedPathTraceTestScene's mesh, but
        // also computes tangents - required for any normal-map test, since
        // PrimitiveIntersect only populates outTangent when
        // Mesh::HasTangents().
        std::unique_ptr<Mesh> MakeTexturedTangentQuadMesh()
        {
            auto mesh = std::make_unique<Mesh>();
            mesh->vertices = {
                Vec3(-0.7f, -0.7f, 0.0f),
                Vec3(0.7f, -0.7f, 0.0f),
                Vec3(0.7f, 0.7f, 0.0f),
                Vec3(-0.7f, 0.7f, 0.0f),
            };
            mesh->uvs = {Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(1.0f, 1.0f), Vec2(0.0f, 1.0f)};
            mesh->indices = {0, 1, 2, 0, 2, 3};
            mesh->CalculateNormals();
            mesh->ComputeTangents();
            mesh->rebuildBVH();
            return mesh;
        }

        // mode: 0 = no normal map, 1 = flat normal map, 2 = tilted normal map.
        void BuildNormalMapTestScene(Scene &scene, int mode)
        {
            Primitive spherePrim;
            spherePrim.type = eSphere;
            spherePrim.sphere.radius = 1.0f;
            spherePrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
            spherePrim.endTransform = spherePrim.startTransform;
            {
                auto sphereMat = std::make_unique<Material>();
                sphereMat->color = Vec3(0.7f, 0.3f, 0.3f);
                sphereMat->roughness = 0.4f;
                spherePrim.materialIndex = scene.AddMaterial(std::move(sphereMat));
            }
            scene.AddPrimitive(spherePrim);

            std::unique_ptr<Mesh> mesh = MakeTexturedTangentQuadMesh();

            Primitive meshPrim;
            meshPrim.type = eMesh;
            meshPrim.startTransform = Transform(Vec3(2.3f, 0.0f, 0.0f));
            meshPrim.endTransform = meshPrim.startTransform;
            meshPrim.mesh = GeometryFromMesh(mesh.get());
            {
                auto meshMat = std::make_unique<Material>();
                meshMat->color = Vec3(0.6f, 0.6f, 0.6f);
                meshMat->roughness = 0.5f;
                if (mode != 0)
                {
                    Texture tex = (mode == 1) ? MakeFlatNormalMapTexture() : MakeTiltedNormalMapTexture();
                    auto texPtr = std::make_unique<Texture>(std::move(tex));
                    int texIndex = scene.AddTexture(std::move(texPtr));
                    meshMat->normalTextureIndex = texIndex;
                }
                meshPrim.materialIndex = scene.AddMaterial(std::move(meshMat));
            }
            meshPrim.mesh.meshIndex = scene.AddMesh(std::move(mesh));
            scene.AddPrimitive(meshPrim);

            Primitive lightPrim;
            lightPrim.type = eSphere;
            lightPrim.sphere.radius = 0.5f;
            lightPrim.startTransform = Transform(Vec3(0.5f, 3.0f, 2.0f));
            lightPrim.endTransform = lightPrim.startTransform;
            {
                auto lightMat = std::make_unique<Material>();
                lightMat->emission = Vec3(8.0f, 8.0f, 7.0f);
                lightPrim.materialIndex = scene.AddMaterial(std::move(lightMat));
            }
            lightPrim.lightSamples = 2;
            scene.AddPrimitive(lightPrim);

            scene.sky.horizon = Vec3(0.6f, 0.7f, 0.9f);
            scene.sky.zenith = Vec3(0.1f, 0.2f, 0.5f);

            scene.Build();
        }
    }

    bool RunNormalMappingValidation()
    {
        bool pass = RunNormalMappingMathValidation();

        // Regression: flat normal map vs no normal map at all, same fixed
        // ray (aimed at the mesh, positioned at x=2.3 per
        // BuildTexturedPathTraceTestScene's precedent) and same per-sample
        // seed across both scenes - only the RNG-driven shading path
        // (BSDF/light sampling) varies sample to sample, so any deviation
        // between the two scenes traces back to the normal-map decode
        // itself, not sampling noise.
        {
            Scene baseline, flat;
            BuildNormalMapTestScene(baseline, 0);
            BuildNormalMapTestScene(flat, 1);

            const Vec3 origin(0.0f, 0.0f, 6.0f);
            const Vec3 meshCenter(2.3f, 0.0f, 0.0f);
            const Vec3 dir = Normalize(meshCenter - origin);

            const int kSampleCount = 16;
            double sumSq = 0.0;
            int bad = 0;
            for (int i = 0; i < kSampleCount; ++i)
            {
                uint32_t seed = (uint32_t)(i * 2246822519u + 1u);

                Random randA(seed);
                Vec3 a = PathTrace(baseline, origin, dir, 1.0f, 3, randA);
                Random randB(seed);
                Vec3 b = PathTrace(flat, origin, dir, 1.0f, 3, randB);

                float d = Length(a - b);
                sumSq += (double)d * d;
                if (d > 1.e-4f)
                    ++bad;
            }
            printf("[M12 NormalMapping] flat-map regression RMSE=%f bad=%d/%d\n", sqrt(sumSq / kSampleCount), bad, kSampleCount);
            pass = pass && (bad == 0);
        }

        bool renderPass = RunPathTraceValidation(
            [](Scene &s) { BuildNormalMapTestScene(s, 2); }, 1.e-2f, "normal-mapped");
        pass = pass && renderPass;

        printf("[M12 NormalMapping] %s\n", pass ? "PASS" : "FAIL");
        return pass;
    }

    // M7 validation-only: CPU-only (no GPU/Vulkan involved - that's M8's job)
    // checks for TextureSampling.h's SampleTextureRGB/SampleTextureR against
    // hand-computed bilinear expected values, plus a textured-mesh render
    // saved as PNG for visual inspection. No file I/O either - the test
    // texture is built directly in memory so this doesn't depend on
    // TextureLoader.h/stb_image working correctly, just the sampling math.
    Texture MakeTest2x2Texture()
    {
        Texture tex;
        tex.width = 2;
        tex.height = 2;
        tex.depth = 1;
        tex.data = new float[2 * 2 * 4];
        auto setTexel = [&](int x, int y, float r, float g, float b, float a) {
            int idx = (y * tex.width + x) * 4;
            tex.data[idx + 0] = r;
            tex.data[idx + 1] = g;
            tex.data[idx + 2] = b;
            tex.data[idx + 3] = a;
        };
        setTexel(0, 0, 1.0f, 0.0f, 0.0f, 1.0f); // red
        setTexel(1, 0, 0.0f, 1.0f, 0.0f, 1.0f); // green
        setTexel(0, 1, 0.0f, 0.0f, 1.0f, 1.0f); // blue
        setTexel(1, 1, 1.0f, 1.0f, 1.0f, 1.0f); // white
        return tex;
    }

    bool RunTextureSamplingValidation()
    {
        bool pass = true;
        auto check = [&](const char *name, Vec3 got, Vec3 expected) {
            float d = Length(got - expected);
            bool ok = d < 1.e-5f;
            printf("[M7 TextureSampling] %s: expected (%.4f,%.4f,%.4f) got (%.4f,%.4f,%.4f) %s\n",
                   name, expected.x, expected.y, expected.z, got.x, got.y, got.z, ok ? "OK" : "MISMATCH");
            pass = pass && ok;
        };

        Texture tex = MakeTest2x2Texture();

        // Sampling exactly at a texel center returns that texel exactly (no
        // interpolation) - the half-texel-centered convention SampleBilinear
        // uses (fx = u*width - 0.5) is exactly what makes this true.
        check("texel(0,0) center", SampleTextureRGB(tex, Vec2(0.25f, 0.25f)), Vec3(1.0f, 0.0f, 0.0f));
        check("texel(1,0) center", SampleTextureRGB(tex, Vec2(0.75f, 0.25f)), Vec3(0.0f, 1.0f, 0.0f));
        check("texel(0,1) center", SampleTextureRGB(tex, Vec2(0.25f, 0.75f)), Vec3(0.0f, 0.0f, 1.0f));
        check("texel(1,1) center", SampleTextureRGB(tex, Vec2(0.75f, 0.75f)), Vec3(1.0f, 1.0f, 1.0f));

        // Dead center of the whole texture is equidistant from all 4 texel
        // centers - bilinear interpolation there is exactly their average.
        Vec3 avgOfFour = (Vec3(1, 0, 0) + Vec3(0, 1, 0) + Vec3(0, 0, 1) + Vec3(1, 1, 1)) * 0.25f;
        check("texture center (avg of 4)", SampleTextureRGB(tex, Vec2(0.5f, 0.5f)), avgOfFour);

        // Wrap-addressing property check (not a hand-computed absolute value,
        // but a hand-reasoned invariant): u=0 and u=1 are the same physical
        // location on a periodic texture, so they must sample identically -
        // same for v. This exercises the wrap path (SampleBilinear's fx=-0.5
        // case) without needing to hand-derive its exact blended color.
        {
            Vec3 atU0 = SampleTextureRGB(tex, Vec2(0.0f, 0.3f));
            Vec3 atU1 = SampleTextureRGB(tex, Vec2(1.0f, 0.3f));
            check("wrap U periodicity (u=0 vs u=1)", atU1, atU0);

            Vec3 atV0 = SampleTextureRGB(tex, Vec2(0.6f, 0.0f));
            Vec3 atV1 = SampleTextureRGB(tex, Vec2(0.6f, 1.0f));
            check("wrap V periodicity (v=0 vs v=1)", atV1, atV0);
        }

        // Single-channel sampling (roughness/metallic path) - same texture,
        // just reading .r instead of .rgb.
        float r00 = SampleTextureR(tex, Vec2(0.25f, 0.25f));
        bool rOk = fabsf(r00 - 1.0f) < 1.e-5f;
        printf("[M7 TextureSampling] SampleTextureR at texel(0,0): expected 1.0000 got %.4f %s\n", r00, rOk ? "OK" : "MISMATCH");
        pass = pass && rOk;

        // M8: GPU-side isolated comparison via TextureSampleTestMain.slang -
        // Material_SampleAlbedo/Roughness/Metallic through the real
        // g_Textures[] array binding, at the same UV grid as the CPU checks
        // above (cross-validated against SampleTextureRGB/SampleTextureR
        // rather than re-deriving expected values by hand - those are already
        // hand-verified above; this is "does the GPU path agree with the
        // now-trusted CPU path", the same relationship BsdfTestMain/
        // ProbeTestMain's validation has to their CPU counterparts).
        {
            Scene scene;
            auto texPtr = std::make_unique<Texture>(MakeTest2x2Texture());
            int texIndex = scene.AddTexture(std::move(texPtr));

            // ToGpuMaterial() always sets the texture-index fields to -1 (it
            // feeds the texture-agnostic BSDF unit test) - set them for real
            // here, after conversion, since this test specifically wants them
            // populated. `mat`'s color/roughness/metallic are deliberately
            // different from the texture's, to prove the override actually
            // fires rather than coincidentally matching the fallback scalar.
            auto mat = std::make_unique<Material>();
            mat->color = Vec3(0.1f, 0.1f, 0.1f);
            mat->roughness = 0.9f;
            mat->metallic = 0.9f;
            GpuMaterial gpuMat = ToGpuMaterial(*mat);
            gpuMat.albedoTextureIndex = texIndex;
            gpuMat.roughnessTextureIndex = texIndex;
            gpuMat.metallicTextureIndex = texIndex;
            gpuMat.opacityTextureIndex = texIndex;

            std::vector<Vec2> testUVs = {
                Vec2(0.25f, 0.25f), Vec2(0.75f, 0.25f), Vec2(0.25f, 0.75f), Vec2(0.75f, 0.75f),
                Vec2(0.5f, 0.5f), Vec2(0.0f, 0.3f), Vec2(1.0f, 0.3f), Vec2(0.6f, 0.0f), Vec2(0.6f, 1.0f),
            };

            std::vector<GpuTextureSampleTestCase> gpuCases;
            gpuCases.reserve(testUVs.size());
            for (auto &uv : testUVs)
            {
                GpuTextureSampleTestCase gc{};
                gc.materialIndex = 0;
                gc.u = uv.x; gc.v = uv.y;
                gpuCases.push_back(gc);
            }

            std::vector<GpuTextureSampleTestResult> gpuResults(testUVs.size());
            bool gpuPass = true;
            {
                std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
                if (!gpuRenderer->IsAvailable())
                {
                    fprintf(stderr, "[M8 TextureSampling GPU] SKIP: VulkanRenderer not available\n");
                    gpuPass = false;
                }
                else
                {
                    std::vector<GpuMaterial> gpuMats = {gpuMat};
                    bool ran = gpuRenderer->RunTextureSampleTest(gpuMats.data(), (int)gpuMats.size(),
                                                                  gpuCases.data(), gpuResults.data(), (int)testUVs.size());
                    if (!ran)
                    {
                        fprintf(stderr, "[M8 TextureSampling GPU] FAIL: dispatch did not run\n");
                        gpuPass = false;
                    }
                }
            }

            if (gpuPass)
            {
                // Tolerance measured, not assumed - see the plan's risk
                // register item on this exact point. The 4 non-texel-center
                // cases below (the wrap-boundary UVs, where tx or ty == 0.5)
                // showed a small, consistent deviation (~0.0016, e.g. an
                // expected 0.1 came back as 0.1015625) traced to GPU hardware
                // bilinear filtering's quantized subtexel interpolation
                // weights (1/256 steps is the classic 8-bit hardware
                // precision - 0.1015625 == 26/256, an exact match). This is
                // not present at texel centers (tx/ty == 0, no interpolation
                // needed) - only where real blending happens. 0.01 is a
                // generous multiple of that ~0.0039 quantization step.
                const float kTolerance = 0.01f;

                double sumSq = 0.0;
                int bad = 0;
                for (size_t i = 0; i < testUVs.size(); ++i)
                {
                    Vec3 cpuAlbedo = SampleTextureRGB(tex, testUVs[i]);
                    float cpuRoughness = SampleTextureR(tex, testUVs[i]);
                    float cpuMetallic = cpuRoughness; // same texture bound to both
                    float cpuOpacity = SampleTextureAlpha(tex, testUVs[i]);

                    const GpuTextureSampleTestResult &gr = gpuResults[i];
                    Vec3 gpuAlbedo(gr.albedoR, gr.albedoG, gr.albedoB);

                    float dAlbedo = Length(cpuAlbedo - gpuAlbedo);
                    float dRoughness = fabsf(cpuRoughness - gr.roughness);
                    float dMetallic = fabsf(cpuMetallic - gr.metallic);
                    float dOpacity = fabsf(cpuOpacity - gr.opacity);

                    sumSq += (double)dAlbedo * dAlbedo;
                    if (dAlbedo > kTolerance || dRoughness > kTolerance || dMetallic > kTolerance || dOpacity > kTolerance)
                    {
                        ++bad;
                        printf("[M8 TextureSampling GPU]   MISMATCH uv=(%.2f,%.2f) cpu=(%.4f,%.4f,%.4f) gpu=(%.4f,%.4f,%.4f) "
                               "d=%.6f cpuOpacity=%.4f gpuOpacity=%.4f dOpacity=%.6f\n",
                               testUVs[i].x, testUVs[i].y, cpuAlbedo.x, cpuAlbedo.y, cpuAlbedo.z,
                               gpuAlbedo.x, gpuAlbedo.y, gpuAlbedo.z, dAlbedo, cpuOpacity, gr.opacity, dOpacity);
                    }
                }
                printf("[M8 TextureSampling GPU] albedo RMSE=%f bad=%d/%zu\n",
                       sqrt(sumSq / testUVs.size()), bad, testUVs.size());
                gpuPass = bad == 0;
            }
            printf("[M8 TextureSampling GPU] %s\n", gpuPass ? "PASS" : "FAIL");
            pass = pass && gpuPass;
        }

        printf("[M7 TextureSampling] %s\n", pass ? "PASS" : "FAIL");

        // Visual smoke test: a UV-textured quad, lit and CPU-path-traced at a
        // single sample (CpuRenderer::Render() has no internal sample loop -
        // see Renderer.cpp - so this is inherently noisy; that's fine, this
        // is "does the texture visibly vary across the surface", not a
        // numeric gate). Not part of `pass` - a rendering hiccup here
        // shouldn't fail the whole harness when the math checks above (the
        // actual M7 correctness gate) already passed or failed on their own.
        {
            const int width = 96, height = 96;
            Scene scene;

            auto texPtr = std::make_unique<Texture>(MakeTest2x2Texture());
            int texIndex = scene.AddTexture(std::move(texPtr));

            auto mesh = std::make_unique<Mesh>();
            mesh->vertices = {
                Vec3(-1.0f, -1.0f, 0.0f),
                Vec3(1.0f, -1.0f, 0.0f),
                Vec3(1.0f, 1.0f, 0.0f),
                Vec3(-1.0f, 1.0f, 0.0f),
            };
            mesh->uvs = {Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(1.0f, 1.0f), Vec2(0.0f, 1.0f)};
            mesh->indices = {0, 1, 2, 0, 2, 3};
            mesh->CalculateNormals();
            mesh->rebuildBVH();

            Primitive quadPrim;
            quadPrim.type = eMesh;
            quadPrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
            quadPrim.endTransform = quadPrim.startTransform;
            quadPrim.mesh = GeometryFromMesh(mesh.get());
            {
                auto mat = std::make_unique<Material>();
                mat->roughness = 0.8f;
                mat->albedoTextureIndex = texIndex;
                quadPrim.materialIndex = scene.AddMaterial(std::move(mat));
            }
            quadPrim.mesh.meshIndex = scene.AddMesh(std::move(mesh));
            scene.AddPrimitive(quadPrim);

            Primitive lightPrim;
            lightPrim.type = eSphere;
            lightPrim.sphere.radius = 0.5f;
            lightPrim.startTransform = Transform(Vec3(0.0f, 2.0f, 3.0f));
            lightPrim.endTransform = lightPrim.startTransform;
            {
                auto lightMat = std::make_unique<Material>();
                lightMat->emission = Vec3(10.0f, 10.0f, 10.0f);
                lightPrim.materialIndex = scene.AddMaterial(std::move(lightMat));
            }
            lightPrim.lightSamples = 1;
            scene.AddPrimitive(lightPrim);

            scene.sky.horizon = Vec3(0.3f, 0.3f, 0.35f);
            scene.sky.zenith = Vec3(0.1f, 0.1f, 0.15f);
            scene.Build();

            Camera camera;
            camera.position = Vec3(0.0f, 0.0f, 3.0f);
            camera.fov = DegToRad(50.0f);

            Options options{};
            options.mode = ePathTrace;
            options.width = width;
            options.height = height;
            options.enableDOF = false;
            options.maxDepth = 3;
            options.maxSamples = 1;
            options.exposure = 1.0f;
            options.limit = 1000.0f;
            options.clamp = 20.0f;

            std::vector<Color> image(width * height, Color(0.0f));
            std::unique_ptr<Renderer> cpuRenderer(CreateCpuRenderer(&scene));
            cpuRenderer->Init(width, height);
            cpuRenderer->Render(camera, options, image.data());

            double sum = 0.0, sumSq = 0.0;
            for (auto &c : image)
            {
                double lum = (double)(c.x + c.y + c.z);
                sum += lum;
                sumSq += lum * lum;
            }
            double n = (double)(width * height);
            double mean = sum / n;
            double variance = sumSq / n - mean * mean;

            WritePng("kernel_validate_texture_cpu.png", width, height, image);
            printf("[M7 TextureSampling] textured quad CPU render: mean luminance=%f variance=%f "
                   "(image written to kernel_validate_texture_cpu.png for visual inspection; "
                   "variance > 0 is expected evidence the texture varies spatially across the quad)\n",
                   mean, variance);
        }

        return pass;
    }

    // M8: a third RunPathTraceValidation() scene variant (alongside the
    // existing analytic-sky/probe-sky pair) exercising a textured mesh -
    // proves the UV-threading + texture-override plumbing through the *full*
    // integrator (BVH traversal, BSDF, NEE/MIS), not just the isolated
    // Material_SampleAlbedo/Roughness/Metallic calls RunTextureSamplingValidation()
    // above already validated. Mirrors BuildPathTraceTestScene's shape (sphere
    // + mesh + emissive-sphere light, analytic sky) with one change: the
    // mesh's material has a texture-driven albedo instead of a flat color, and
    // the mesh carries real per-vertex UVs (BuildPathTraceTestScene's shared
    // MakeQuadMesh() deliberately left untouched - no UVs - so M1/M3's own
    // results stay byte-for-byte what they were before this milestone).
    void BuildTexturedPathTraceTestScene(Scene &scene)
    {
        Primitive spherePrim;
        spherePrim.type = eSphere;
        spherePrim.sphere.radius = 1.0f;
        spherePrim.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
        spherePrim.endTransform = spherePrim.startTransform;
        {
            auto sphereMat = std::make_unique<Material>();
            sphereMat->color = Vec3(0.7f, 0.3f, 0.3f);
            sphereMat->roughness = 0.4f;
            spherePrim.materialIndex = scene.AddMaterial(std::move(sphereMat));
        }
        scene.AddPrimitive(spherePrim);

        auto mesh = std::make_unique<Mesh>();
        mesh->vertices = {
            Vec3(-0.7f, -0.7f, 0.0f),
            Vec3(0.7f, -0.7f, 0.0f),
            Vec3(0.7f, 0.7f, 0.0f),
            Vec3(-0.7f, 0.7f, 0.0f),
        };
        mesh->uvs = {Vec2(0.0f, 0.0f), Vec2(1.0f, 0.0f), Vec2(1.0f, 1.0f), Vec2(0.0f, 1.0f)};
        mesh->indices = {0, 1, 2, 0, 2, 3};
        mesh->CalculateNormals();
        mesh->rebuildBVH();

        Primitive meshPrim;
        meshPrim.type = eMesh;
        meshPrim.startTransform = Transform(Vec3(2.3f, 0.0f, 0.0f));
        meshPrim.endTransform = meshPrim.startTransform;
        meshPrim.mesh = GeometryFromMesh(mesh.get());
        {
            auto texPtr = std::make_unique<Texture>(MakeTest2x2Texture());
            int texIndex = scene.AddTexture(std::move(texPtr));

            // Deliberately different from the texture's own colors, so a
            // mismatch between "override fired" and "override silently
            // no-op'd, fell back to the scalar" is visible in the result,
            // not accidentally masked by a coincidental match.
            auto meshMat = std::make_unique<Material>();
            meshMat->color = Vec3(0.9f, 0.9f, 0.9f);
            meshMat->roughness = 0.6f;
            meshMat->albedoTextureIndex = texIndex;
            meshPrim.materialIndex = scene.AddMaterial(std::move(meshMat));
        }
        meshPrim.mesh.meshIndex = scene.AddMesh(std::move(mesh));
        scene.AddPrimitive(meshPrim);

        Primitive lightPrim;
        lightPrim.type = eSphere;
        lightPrim.sphere.radius = 0.5f;
        lightPrim.startTransform = Transform(Vec3(0.0f, 3.0f, 1.0f));
        lightPrim.endTransform = lightPrim.startTransform;
        {
            auto lightMat = std::make_unique<Material>();
            lightMat->emission = Vec3(8.0f, 8.0f, 7.0f);
            lightPrim.materialIndex = scene.AddMaterial(std::move(lightMat));
        }
        lightPrim.lightSamples = 2;
        scene.AddPrimitive(lightPrim);

        scene.sky.horizon = Vec3(0.6f, 0.7f, 0.9f);
        scene.sky.zenith = Vec3(0.1f, 0.2f, 0.5f);

        scene.Build();
    }

    // M5 validation-only: go/no-go spike for binding an *array* of combined
    // image samplers (Sampler2D g_Textures[N]) via slang-rhi's ShaderCursor -
    // see the plan's "M5 - Texture-array binding spike" section and
    // VulkanRenderer::RunTextureArraySpike(). Uploads 4 distinct 1x1 textures
    // with known solid colors, dispatches TextureArraySpike.slang (which
    // samples g_Textures[i] for each i), and checks the readback matches what
    // was uploaded - no scene/BVH/camera involved.
    bool RunTextureArraySpikeValidation()
    {
        std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(nullptr)));
        if (!gpuRenderer->IsAvailable())
        {
            fprintf(stderr, "FAIL: VulkanRenderer not available on this machine\n");
            return false;
        }

        static const float kExpected[VulkanRenderer::kTextureArraySpikeCount][4] = {
            {1.0f, 0.0f, 0.0f, 1.0f},
            {0.0f, 1.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 1.0f, 1.0f},
            {1.0f, 1.0f, 0.0f, 1.0f},
        };

        GpuTextureArraySpikeResult results[VulkanRenderer::kTextureArraySpikeCount];
        bool ran = gpuRenderer->RunTextureArraySpike(results);
        if (!ran)
        {
            printf("[M5 TextureArraySpike] FAIL (dispatch did not run)\n");
            return false;
        }

        bool pass = true;
        for (uint32_t i = 0; i < VulkanRenderer::kTextureArraySpikeCount; ++i)
        {
            const GpuTextureArraySpikeResult &r = results[i];
            const float *e = kExpected[i];
            bool ok = fabsf(r.r - e[0]) < 1.e-5f && fabsf(r.g - e[1]) < 1.e-5f &&
                      fabsf(r.b - e[2]) < 1.e-5f && fabsf(r.a - e[3]) < 1.e-5f;
            printf("[M5 TextureArraySpike] slot %u: expected (%.2f,%.2f,%.2f,%.2f) got (%.2f,%.2f,%.2f,%.2f) %s\n",
                   i, e[0], e[1], e[2], e[3], r.r, r.g, r.b, r.a, ok ? "OK" : "MISMATCH");
            pass = pass && ok;
        }

        printf("[M5 TextureArraySpike] %s\n", pass ? "PASS" : "FAIL");
        return pass;
    }

    bool RunProbeValidation()
    {
        const int gridW = 32;
        const int gridH = 32;
        const int count = gridW * gridH;

        Scene scene;
        scene.sky.probe = MakeTestProbe(24, 12);

        std::vector<GpuProbeTestCase> gpuCases;
        std::vector<Vec3> dirs;
        std::vector<uint32_t> seeds;
        gpuCases.reserve(count);
        dirs.reserve(count);
        seeds.reserve(count);

        for (int iy = 0; iy < gridH; ++iy)
        {
            for (int ix = 0; ix < gridW; ++ix)
            {
                float theta = kPi * (float(iy) + 0.5f) / float(gridH);
                float phi = 2.0f * kPi * (float(ix) + 0.5f) / float(gridW);
                Vec3 dir = Normalize(Vec3(sinf(theta) * cosf(phi), cosf(theta), sinf(theta) * sinf(phi)));
                uint32_t seed = (uint32_t)((ix * gridH + iy) * 40503u + 1u);

                dirs.push_back(dir);
                seeds.push_back(seed);

                GpuProbeTestCase gc{};
                gc.seed = seed;
                gc.evalDirX = dir.x; gc.evalDirY = dir.y; gc.evalDirZ = dir.z;
                gc.pdfDirX = dir.x; gc.pdfDirY = dir.y; gc.pdfDirZ = dir.z;
                gpuCases.push_back(gc);
            }
        }

        std::vector<GpuProbeTestResult> gpuResults(count);
        {
            std::unique_ptr<VulkanRenderer> gpuRenderer(static_cast<VulkanRenderer *>(CreateVulkanRenderer(&scene)));
            if (!gpuRenderer->IsAvailable())
            {
                fprintf(stderr, "FAIL: VulkanRenderer not available on this machine\n");
                return false;
            }
            gpuRenderer->RunProbeTest(gpuCases.data(), gpuResults.data(), count);
        }

        auto closeEnough = [](float a, float b, float atol, float rtol) {
            return fabsf(a - b) <= atol + rtol * fmaxf(fabsf(a), fabsf(b));
        };

        double sumSqSampleDir = 0.0, sumSqSamplePdf = 0.0, sumSqEval = 0.0, sumSqPdf = 0.0;
        int badSampleDir = 0, badSamplePdf = 0, badEval = 0, badPdf = 0;

        for (int i = 0; i < count; ++i)
        {
            Random rand(seeds[i]);
            Vec3 cpuSampleDir, cpuSampleColor;
            float cpuSamplePdf;
            ProbeSample(scene.sky.probe, cpuSampleDir, cpuSampleColor, cpuSamplePdf, rand);

            Vec3 cpuEvalColor = ProbeEval(scene.sky.probe, ProbeDirToUV(dirs[i]));
            float cpuPdfAtDir = ProbePdf(scene.sky.probe, dirs[i]);

            const GpuProbeTestResult &gr = gpuResults[i];

            float dSampleDir = Length(cpuSampleDir - Vec3(gr.sampleDirX, gr.sampleDirY, gr.sampleDirZ));
            float dEval = Length(cpuEvalColor - Vec3(gr.evalColorX, gr.evalColorY, gr.evalColorZ));

            sumSqSampleDir += dSampleDir * dSampleDir;
            sumSqEval += dEval * dEval;
            sumSqSamplePdf += (cpuSamplePdf - gr.samplePdf) * (double)(cpuSamplePdf - gr.samplePdf);
            sumSqPdf += (cpuPdfAtDir - gr.pdfAtDir) * (double)(cpuPdfAtDir - gr.pdfAtDir);

            if (dSampleDir > 1.e-3f)
                ++badSampleDir;
            // ProbeEval intentionally differs: GPU uses hardware bilinear
            // filtering, CPU uses nearest-texel - a real, pre-approved, small
            // deviation (see the plan's "Texture / bump-map handling" section),
            // so a looser tolerance is used here specifically.
            if (!closeEnough(cpuEvalColor.x, gr.evalColorX, 0.0f, 0.15f) ||
                !closeEnough(cpuEvalColor.y, gr.evalColorY, 0.0f, 0.15f) ||
                !closeEnough(cpuEvalColor.z, gr.evalColorZ, 0.0f, 0.15f))
                ++badEval;
            if (!closeEnough(cpuSamplePdf, gr.samplePdf, 1.e-2f, 2.e-2f))
                ++badSamplePdf;
            if (!closeEnough(cpuPdfAtDir, gr.pdfAtDir, 1.e-2f, 2.e-2f))
                ++badPdf;
        }

        printf("[M4 Probe] sampleDir RMSE=%f eval RMSE=%f samplePdf RMSE=%f pdf RMSE=%f\n",
               sqrt(sumSqSampleDir / count), sqrt(sumSqEval / count), sqrt(sumSqSamplePdf / count), sqrt(sumSqPdf / count));
        printf("[M4 Probe] bad: sampleDir=%d/%d eval(bilinear-vs-nearest)=%d/%d samplePdf=%d/%d pdf=%d/%d\n",
               badSampleDir, count, badEval, count, badSamplePdf, count, badPdf, count);

        // sampleDir/samplePdf/pdf are all deterministic CDF-table lookups and
        // expected to match tightly; eval is the one place a genuine, expected
        // CPU-vs-GPU difference exists (bilinear vs nearest - see above), so it
        // gets its own, much more permissive threshold rather than diluting the
        // other three metrics' bar.
        const int kBadCountThreshold = count / 50;   // ~2%
        const int kEvalBadCountThreshold = count / 10; // ~10%
        bool pass = badSampleDir <= kBadCountThreshold && badEval <= kEvalBadCountThreshold &&
                    badSamplePdf <= kBadCountThreshold && badPdf <= kBadCountThreshold;
        printf("[M4 Probe] %s\n", pass ? "PASS" : "FAIL");
        return pass;
    }
}

int main()
{
    bool normalsPass = RunNormalsValidation();
    bool bsdfPass = RunBsdfValidation();
    bool pathTracePass = RunPathTraceValidation(
        [](Scene &s) { BuildPathTraceTestScene(s, false); }, 1.e-2f, "analytic-sky");
    bool pathTraceProbePass = RunPathTraceValidation(
        [](Scene &s) { BuildPathTraceTestScene(s, true); }, 0.12f, "probe-sky");
    RunPathTraceSmokeTest();
    bool probePass = RunProbeValidation();
    bool textureArraySpikePass = RunTextureArraySpikeValidation();
    bool textureSamplingPass = RunTextureSamplingValidation();
    // atol: same magnitude as M8's isolated g_Textures[] hardware-bilinear-
    // quantization tolerance (RunTextureSamplingValidation's kTolerance),
    // since that's the only new (non-probe) CPU/GPU deviation source this
    // scene introduces relative to the already-tight analytic-sky case.
    bool pathTraceTexturedPass = RunPathTraceValidation(
        [](Scene &s) { BuildTexturedPathTraceTestScene(s); }, 0.01f, "textured-mesh");
    bool motionBlurPass = RunMotionBlurValidation();
    bool instancingPass = RunInstancingValidation();
    bool opacityPass = RunOpacityValidation();
    bool normalMappingPass = RunNormalMappingValidation();
    // Tier-2 "punctual lights + rect/disk area lights" - see
    // BuildPunctualAndAreaLightTestScene's comment. atol matches the tight
    // analytic-sky case above (0.01) since this scene introduces no new
    // known CPU/GPU deviation source (no probe, no textures).
    bool punctualLightsPass = RunPathTraceValidation(
        [](Scene &s) { BuildPunctualAndAreaLightTestScene(s); }, 0.01f, "punctual+area lights");

    bool pass = normalsPass && bsdfPass && pathTracePass && pathTraceProbePass && probePass &&
                textureArraySpikePass && textureSamplingPass && pathTraceTexturedPass && motionBlurPass &&
                instancingPass && opacityPass && normalMappingPass && punctualLightsPass;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
