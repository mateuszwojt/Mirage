// Mirrors the Disney-BSDF-relevant fields of mirage/mirage/shaders/Material.h,
// plus (as of M8) the three texture-index fields added there. The dead
// albedoMap/bumpMap Texture fields are still not ported - bump mapping stays
// dead code (nothing calls EvaluateBumpNormal), unrelated to the new albedo/
// roughness/metallic texture-index fields below. GpuLayout.h's GpuMaterial is
// the byte-identical C++ mirror.

import Constants;

struct GpuMaterial
{
    float3 color;
    float3 emission;
    float3 absorption;

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

    int albedoTextureIndex;
    int roughnessTextureIndex;
    int metallicTextureIndex;
};

// Mirrors Material::GetIndexOfRefraction().
float Material_GetIndexOfRefraction(GpuMaterial mat)
{
    if (mat.eta == 0.0f)
        return 2.0f / (1.0f - sqrt(0.08f * mat.specular)) - 1.0f;
    return mat.eta;
}

// Fixed-size array of per-material textures (see Constants.slang's
// kMaxMaterialTextures) - one shared binding, indexed by GpuMaterial's
// xxxTextureIndex fields, mirroring the mechanism confirmed by the M5 spike
// (TextureArraySpike.slang). Slots beyond the scene's actual texture count are
// bound to a dummy 1x1 white texture by VulkanRenderer::UploadMaterialTextures
// - the pipeline's descriptor layout is fixed at link time regardless of
// runtime scene content, same reasoning as g_ProbeTexture's always-bound dummy
// (see Probe.slang).
uniform Sampler2D g_Textures[kMaxMaterialTextures];

// Mirrors TextureSampling.h's SampleTextureRGB/SampleTextureR - hardware
// bilinear+wrap sampling via Sampler2D.SampleLevel, deliberately matched by
// the CPU's hand-rolled bilinear sampler (not the probe's historical nearest-
// texel CPU convention) so kernel_validate can hold this to a tight bar.
// Falls back to the material's scalar/color field when no texture is bound.
float3 Material_SampleAlbedo(GpuMaterial mat, float2 uv)
{
    if (mat.albedoTextureIndex < 0)
        return mat.color;
    return g_Textures[mat.albedoTextureIndex].SampleLevel(uv, 0.0f).rgb;
}

float Material_SampleRoughness(GpuMaterial mat, float2 uv)
{
    if (mat.roughnessTextureIndex < 0)
        return mat.roughness;
    return g_Textures[mat.roughnessTextureIndex].SampleLevel(uv, 0.0f).r;
}

float Material_SampleMetallic(GpuMaterial mat, float2 uv)
{
    if (mat.metallicTextureIndex < 0)
        return mat.metallic;
    return g_Textures[mat.metallicTextureIndex].SampleLevel(uv, 0.0f).r;
}
