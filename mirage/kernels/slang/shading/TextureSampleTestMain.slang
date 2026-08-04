// M8 validation-only entry point: evaluates Material_SampleAlbedo/
// Material_SampleRoughness/Material_SampleMetallic for a batch of independent
// (material, uv) test cases through the real g_Textures[] array binding (no
// scene/BVH/camera involved - a unit-test kernel for Material.slang's texture
// sampling in isolation, mirroring BsdfTestMain.slang's pattern). Compared
// against the same CPU-side SampleTextureRGB/SampleTextureR calls (via
// TextureSampling.h) in tools/kernel_validate/KernelValidate.cpp.
//
// Deliberately self-contained (own g_Materials, not SceneBuffers.slang's) -
// mirrors TextureArraySpike.slang's independence from a real Scene, and
// avoids having to bind the full scene-buffer set (g_Primitives,
// g_MeshDescriptors, g_SceneBVH, ...) just to test material texture sampling.

import Material;

struct TextureSampleTestCase
{
    int materialIndex;
    float2 uv;
};

struct TextureSampleTestResult
{
    float3 albedo;
    float roughness;
    float metallic;
    float opacity; // Tier-1 "opacity/cutout transparency" - Material_SampleOpacity
};

struct CountParams
{
    uint count;
};

uniform StructuredBuffer<CountParams> g_Frame;
uniform StructuredBuffer<GpuMaterial> g_Materials;
uniform StructuredBuffer<TextureSampleTestCase> g_Cases;
uniform RWStructuredBuffer<TextureSampleTestResult> g_Results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= g_Frame[0].count)
        return;

    TextureSampleTestCase tc = g_Cases[i];
    GpuMaterial mat = g_Materials[tc.materialIndex];

    TextureSampleTestResult result;
    result.albedo = Material_SampleAlbedo(mat, tc.uv);
    result.roughness = Material_SampleRoughness(mat, tc.uv);
    result.metallic = Material_SampleMetallic(mat, tc.uv);
    result.opacity = Material_SampleOpacity(mat, tc.uv);

    g_Results[i] = result;
}
