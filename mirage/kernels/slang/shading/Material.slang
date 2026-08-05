// Mirrors the Disney-BSDF-relevant fields of mirage/mirage/shaders/Material.h,
// plus (as of M8) the three texture-index fields added there, plus (Tier-1
// "opacity/cutout transparency") opacity/opacityTextureIndex. The dead
// albedoMap/bumpMap Texture fields are still not ported - bump mapping stays
// dead code (nothing calls EvaluateBumpNormal), unrelated to the texture-index
// fields below. GpuLayout.h's GpuMaterial is the byte-identical C++ mirror.

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
    float opacity; // 1.0 = fully opaque (default) - see TraceWithCutout in Trace.slang

    int albedoTextureIndex;
    int roughnessTextureIndex;
    int metallicTextureIndex;
    int opacityTextureIndex; // -1 = none, use `opacity` scalar as-is
    int normalTextureIndex;  // -1 = none, no normal perturbation - see NormalMapping.slang
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

// Mirrors TextureSampling.h's SampleTextureAlpha - used for opacity/cutout
// transparency (see Trace.slang's TraceWithCutout). Unlike the other
// Material_Sample* functions above, there is no "falls back to a scalar
// when no texture is bound" case to worry about here at the call site: this
// is only ever called after already checking opacityTextureIndex >= 0.
float Material_SampleOpacity(GpuMaterial mat, float2 uv)
{
    if (mat.opacityTextureIndex < 0)
        return mat.opacity;
    return g_Textures[mat.opacityTextureIndex].SampleLevel(uv, 0.0f).a;
}

// Raw [0,1] RGB texel from the bound normal-map texture - the decode/TBN
// transform live in NormalMapping.slang's ApplyNormalMap, not here, mirroring
// how the CPU side keeps SampleTextureRGB (texture sampling) and
// ApplyNormalMap (the math) in separate files. Only ever called after
// already checking normalTextureIndex >= 0, same convention as
// Material_SampleOpacity above.
float3 Material_SampleNormal(GpuMaterial mat, float2 uv)
{
    return g_Textures[mat.normalTextureIndex].SampleLevel(uv, 0.0f).rgb;
}
