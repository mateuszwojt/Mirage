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

// Per-slot width/height/UDIM-grid metadata (Tier-2 "mipmapping"/"UDIM"),
// parallel to g_Textures[] above and indexed the same way - mirrors
// GpuLayout.h's GpuTextureInfo. Defined here (not scene/SceneBuffers.slang,
// which already `import Material` for GpuMaterial) to avoid a circular
// import.
struct GpuTextureInfo
{
    int width;
    int height;
    int udimGridWidth;
    int udimGridHeight;
};

uniform StructuredBuffer<GpuTextureInfo> g_TextureInfo;

// Mirrors TextureSampling.h's UdimAtlasUV exactly (see that function's
// comment for the tile-placement/row-order convention) - a no-op for an
// ordinary single-tile texture (udimGridWidth == udimGridHeight == 1),
// otherwise remaps uv's integer part (tile index) and fractional part
// (local UV) into the composited atlas texture's own [0,1] space. See
// SceneRenderer.cpp's LoadUdimAtlas for how the atlas itself is built.
float2 UdimAtlasUV(GpuTextureInfo info, float2 uv)
{
    if (info.udimGridWidth <= 1 && info.udimGridHeight <= 1)
        return uv;

    float tileU = floor(uv.x);
    float tileV = floor(uv.y);
    float localU = uv.x - tileU;
    float localV = uv.y - tileV;

    float clampedU = clamp(tileU, 0.0f, float(info.udimGridWidth - 1));
    float clampedV = clamp(tileV, 0.0f, float(info.udimGridHeight - 1));

    return float2((clampedU + localU) / float(info.udimGridWidth),
                  (clampedV + localV) / float(info.udimGridHeight));
}

// Mirrors TextureSampling.h's SampleTextureRGB/SampleTextureR - hardware
// bilinear+wrap sampling via Sampler2D.SampleLevel, deliberately matched by
// the CPU's hand-rolled bilinear sampler (not the probe's historical nearest-
// texel CPU convention) so kernel_validate can hold this to a tight bar.
// Falls back to the material's scalar/color field when no texture is bound.
// mipLevel is a required parameter (not defaulted - this codebase's Slang
// files have no default-parameter precedent, matching the explicit-argument
// convention every other threaded value here uses, e.g. numScenePrims/
// numLights) - callers with no mip heuristic available (e.g.
// TextureSampleTestMain.slang's isolated function tests) pass 0.0f
// explicitly, same as the CPU-side default. .SampleLevel() itself does
// hardware trilinear filtering between adjacent integer levels when the
// bound texture has mips (see VulkanRenderer::UploadMaterialTextures),
// unlike the CPU's hand-rolled SampleTrilinear which blends two bilinear
// samples explicitly - same numerical intent, different mechanism, since the
// GPU gets this for free from the sampler hardware.
float3 Material_SampleAlbedo(GpuMaterial mat, float2 uv, float mipLevel)
{
    if (mat.albedoTextureIndex < 0)
        return mat.color;
    float2 remappedUV = UdimAtlasUV(g_TextureInfo[mat.albedoTextureIndex], uv);
    return g_Textures[mat.albedoTextureIndex].SampleLevel(remappedUV, mipLevel).rgb;
}

float Material_SampleRoughness(GpuMaterial mat, float2 uv, float mipLevel)
{
    if (mat.roughnessTextureIndex < 0)
        return mat.roughness;
    float2 remappedUV = UdimAtlasUV(g_TextureInfo[mat.roughnessTextureIndex], uv);
    return g_Textures[mat.roughnessTextureIndex].SampleLevel(remappedUV, mipLevel).r;
}

float Material_SampleMetallic(GpuMaterial mat, float2 uv, float mipLevel)
{
    if (mat.metallicTextureIndex < 0)
        return mat.metallic;
    float2 remappedUV = UdimAtlasUV(g_TextureInfo[mat.metallicTextureIndex], uv);
    return g_Textures[mat.metallicTextureIndex].SampleLevel(remappedUV, mipLevel).r;
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
    float2 remappedUV = UdimAtlasUV(g_TextureInfo[mat.opacityTextureIndex], uv);
    return g_Textures[mat.opacityTextureIndex].SampleLevel(remappedUV, 0.0f).a;
}

// Raw [0,1] RGB texel from the bound normal-map texture - the decode/TBN
// transform live in NormalMapping.slang's ApplyNormalMap, not here, mirroring
// how the CPU side keeps SampleTextureRGB (texture sampling) and
// ApplyNormalMap (the math) in separate files. Only ever called after
// already checking normalTextureIndex >= 0, same convention as
// Material_SampleOpacity above.
float3 Material_SampleNormal(GpuMaterial mat, float2 uv)
{
    float2 remappedUV = UdimAtlasUV(g_TextureInfo[mat.normalTextureIndex], uv);
    return g_Textures[mat.normalTextureIndex].SampleLevel(remappedUV, 0.0f).rgb;
}
