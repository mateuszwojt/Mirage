#pragma once

// C++ mirror structs for every GPU-side Slang buffer/uniform struct used by the
// Vulkan kernel path. Kept byte-for-byte identical to their Slang counterparts
// (see kernels/slang/geometry/*.slang, kernels/slang/sampling/CameraSampler.slang)
// under scalar buffer layout (VK_EXT_scalar_block_layout, confirmed supported on
// the M0 dev machine) - no std430 padding, so uploads are plain memcpy.
//
// Manual mirroring, no codegen - matches the existing BVHNode precedent
// (mirage/mirage/core/BVH.h). static_assert(sizeof(...)) catches size drift; it
// cannot catch bitfield-order drift (see GpuBVHNode below), which is why
// kernel_validate additionally does a runtime round-trip check.

#include <cstdint>
#include <cstddef>

namespace Mirage
{
    // Mirrors Transform (math/Transform.h): Vec3 p; Quat r; float s.
    struct GpuTransform
    {
        float px, py, pz;
        float rx, ry, rz, rw;
        float s;
    };
    static_assert(sizeof(GpuTransform) == 32, "GpuTransform must match Transform's 32-byte layout");

    // Mirrors BVHNode (core/BVH.h): Bounds bounds; unsigned leftIndex;
    // unsigned rightIndex:31; unsigned leaf:1. The bitfield packing is
    // implementation-defined C++ - kernel_validate's VerifyBVHNodeLayout() below
    // does a runtime check against the actual toolchain's layout, this
    // static_assert only catches gross size drift.
    struct GpuBVHNode
    {
        float lowerX, lowerY, lowerZ;
        float upperX, upperY, upperZ;
        uint32_t leftIndex;
        uint32_t rightIndexAndLeaf; // bits [0:30] = rightIndex, bit 31 = leaf
    };
    static_assert(sizeof(GpuBVHNode) == 32, "GpuBVHNode must match BVHNode's 32-byte layout");

    // Mirrors prims/Primitive.h's Primitive. unionData holds sphere radius (.x)
    // or plane equation (xyzw); meshIndex indexes GpuMeshDescriptor when
    // type == eMesh; materialIndex indexes g_Materials (one GpuMaterial per
    // primitive - no dedup, matching one-material-per-Primitive on the CPU side).
    struct GpuPrimitive
    {
        GpuTransform startTransform;
        GpuTransform endTransform;
        float unionDataX, unionDataY, unionDataZ, unionDataW;
        int32_t type; // matches Mirage::Type (eSphere=0, ePlane=1, eMesh=2)
        int32_t meshIndex; // -1 if type != eMesh
        int32_t lightSamples;
        int32_t materialIndex;
        int32_t hydraId; // mirrors Primitive::hydraId - stable id for the primId AOV
    };
    static_assert(sizeof(GpuPrimitive) == 32 + 32 + 16 + 16 + 4, "GpuPrimitive layout drifted");

    // Mirrors prims/Primitive.h's MeshGeometry, with raw pointers replaced by
    // offsets into the scene's mega-buffers (see scene/SceneBuffers.slang).
    // hasUVs distinguishes "this mesh has no UV data" from "its UVs start at
    // buffer offset 0" (a legitimate real offset for the first mesh uploaded) -
    // uvOffset alone can't disambiguate those two cases.
    struct GpuMeshDescriptor
    {
        uint32_t vertexOffset, indexOffset, uvOffset, nodeOffset;
        uint32_t numVertices, numIndices, numNodes;
        float area;
        uint32_t hasUVs;
    };
    static_assert(sizeof(GpuMeshDescriptor) == 36, "GpuMeshDescriptor layout drifted");

    // Precomputed on the host from Mirage::Camera + Options (mirrors
    // utils/Util.h's CameraSampler constructor output) - avoids porting
    // quaternion camera math to the GPU. rasterToWorld is column-major,
    // matching math/Mat44.h's existing storage (cols[4][4]) byte-for-byte.
    struct GpuCamera
    {
        float rasterToWorld[16];
        float originX, originY, originZ;
        float aperture;
        float focalPoint;
        uint32_t enableDOF;
        float shutterStart, shutterEnd;
    };
    static_assert(sizeof(GpuCamera) == 64 + 12 + 4 + 4 + 4 + 8, "GpuCamera layout drifted");

    // Per-dispatch uniform. sampleIndex/frameSeed drive PathTraceMain.slang's
    // per-pixel RNG seeding for progressive accumulation (see the plan's "Path
    // integrator kernel design" section) - intentionally independent of the
    // CPU's single shared sequential Random stream, so per-pixel noise patterns
    // are not expected to match between CPU and GPU renders (only the BSDF/
    // PathTrace *functions* are validated bit-close, via explicitly-seeded unit
    // tests - see GpuPathTraceTestCase below).
    struct GpuFrameParams
    {
        uint32_t width;
        uint32_t height;
        uint32_t numPrimitives;
        int32_t maxDepth;
        uint32_t frameSeed;
        uint32_t sampleIndex;
        float clampValue;
    };
    static_assert(sizeof(GpuFrameParams) == 28, "GpuFrameParams layout drifted");

    // Mirrors the Disney-BSDF-relevant fields of shaders/Material.h, plus (as
    // of M8) the three texture-index fields added there. See
    // kernels/slang/shading/Material.slang for the byte-identical Slang struct.
    // The texture indices are appended, not interleaved - existing field
    // offsets are unchanged.
    struct GpuMaterial
    {
        float colorX, colorY, colorZ;
        float emissionX, emissionY, emissionZ;
        float absorptionX, absorptionY, absorptionZ;

        float eta;
        float metallic;
        float subsurface;
        float specular;
        float roughness;
        float specularTint;
        float anisotropic;
        float sheen;
        float sheenTint;
        float clearcoat;
        float clearcoatGloss;
        float transmission;

        int32_t albedoTextureIndex;    // -1 = none, use color as-is
        int32_t roughnessTextureIndex; // -1 = none, use roughness as-is
        int32_t metallicTextureIndex;  // -1 = none, use metallic as-is
    };
    static_assert(sizeof(GpuMaterial) == 36 + 48 + 12, "GpuMaterial layout drifted");

    // M2 validation-only: one BSDF Eval/Pdf/Sample test case, mirrors
    // kernels/slang/shading/BsdfTestMain.slang's BsdfTestCase.
    struct GpuBsdfTestCase
    {
        GpuMaterial material;
        float etaI, etaO;
        float nx, ny, nz;
        float vx, vy, vz;
        float lx, ly, lz;
        uint32_t seed;
    };
    static_assert(sizeof(GpuBsdfTestCase) == 96 + 8 + 36 + 4, "GpuBsdfTestCase layout drifted");

    // M2 validation-only: BSDF Eval/Pdf/Sample results for one test case.
    struct GpuBsdfTestResult
    {
        float evalX, evalY, evalZ;
        float pdf;
        float sampleDirX, sampleDirY, sampleDirZ;
        float samplePdf;
        int32_t sampleType;
    };
    static_assert(sizeof(GpuBsdfTestResult) == 36, "GpuBsdfTestResult layout drifted");

    // Mirrors core/Scene.h's Sky: horizon/zenith analytic gradient, plus
    // probeValid/probeWidth/probeHeight for the HDR environment path (see
    // sampling/Probe.slang; probe pixel/CDF/PDF data is uploaded as separate
    // buffers/texture, not part of this struct, mirroring Probe's own
    // width/height + separate data/cdf/pdf pointers).
    struct GpuSky
    {
        float horizonX, horizonY, horizonZ;
        float zenithX, zenithY, zenithZ;
        int32_t probeValid;
        int32_t probeWidth;
        int32_t probeHeight;
    };
    static_assert(sizeof(GpuSky) == 36, "GpuSky layout drifted");

    // M3 validation-only: one PathTrace test case - an explicitly-seeded,
    // independent primary ray through the shared scene (mirrors
    // kernels/slang/integrator/PathTraceTestMain.slang's PathTraceTestCase).
    // Explicit seeding (not the per-pixel frameSeed/sampleIndex hash
    // PathTraceMain.slang uses for real rendering) is what makes a tight
    // CPU-vs-GPU comparison possible here - see the plan's note that
    // per-pixel-hashed RNG streams intentionally differ between the CPU and GPU
    // renderers and are not meant to match bit-for-bit.
    struct GpuPathTraceTestCase
    {
        float originX, originY, originZ;
        float dirX, dirY, dirZ;
        float time;
        int32_t maxDepth;
        uint32_t seed;
        uint32_t numPrimitives;
    };
    static_assert(sizeof(GpuPathTraceTestCase) == 40, "GpuPathTraceTestCase layout drifted");

    struct GpuPathTraceTestResult
    {
        float radianceX, radianceY, radianceZ;
    };
    static_assert(sizeof(GpuPathTraceTestResult) == 12, "GpuPathTraceTestResult layout drifted");

    // M4 validation-only: one Probe Sample/Eval/Pdf test case, mirrors
    // kernels/slang/shading/ProbeTestMain.slang's ProbeTestCase.
    struct GpuProbeTestCase
    {
        uint32_t seed;
        float evalDirX, evalDirY, evalDirZ;
        float pdfDirX, pdfDirY, pdfDirZ;
    };
    static_assert(sizeof(GpuProbeTestCase) == 28, "GpuProbeTestCase layout drifted");

    struct GpuProbeTestResult
    {
        float sampleDirX, sampleDirY, sampleDirZ;
        float sampleColorX, sampleColorY, sampleColorZ;
        float samplePdf;
        float evalColorX, evalColorY, evalColorZ;
        float pdfAtDir;
    };
    static_assert(sizeof(GpuProbeTestResult) == 44, "GpuProbeTestResult layout drifted");

    // M5 validation-only: one sampled color from TextureArraySpike.slang's
    // g_Textures[i].SampleLevel() - see VulkanRenderer::RunTextureArraySpike().
    struct GpuTextureArraySpikeResult
    {
        float r, g, b, a;
    };
    static_assert(sizeof(GpuTextureArraySpikeResult) == 16, "GpuTextureArraySpikeResult layout drifted");

    // M8 validation-only: one Material_SampleAlbedo/Roughness/Metallic test
    // case - mirrors kernels/slang/shading/TextureSampleTestMain.slang's
    // TextureSampleTestCase. Isolated function-call comparison (no scene/BVH/
    // camera involved), same pattern as GpuBsdfTestCase - materialIndex
    // indexes g_Materials (uploaded from the test's Scene), uv is the sample
    // coordinate.
    struct GpuTextureSampleTestCase
    {
        int32_t materialIndex;
        float u, v;
    };
    static_assert(sizeof(GpuTextureSampleTestCase) == 12, "GpuTextureSampleTestCase layout drifted");

    struct GpuTextureSampleTestResult
    {
        float albedoR, albedoG, albedoB;
        float roughness;
        float metallic;
    };
    static_assert(sizeof(GpuTextureSampleTestResult) == 20, "GpuTextureSampleTestResult layout drifted");
}
