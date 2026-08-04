// M5 validation-only kernel: spikes binding an *array* of combined image
// samplers (`Sampler2D g_Textures[N]`) via slang-rhi's ShaderCursor, before
// the real per-material texture array (Material.slang's g_Textures) is built
// against the assumption that this works. See the plan's "M5 - Texture-array
// binding spike" section. Every existing texture binding in this codebase
// (g_ProbeTexture) is a single resource, not an array element, so this has no
// precedent here to copy from.

static const uint kSpikeTextureCount = 4;

uniform Sampler2D g_Textures[kSpikeTextureCount];
uniform RWStructuredBuffer<float4> g_Output;

[shader("compute")]
[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= kSpikeTextureCount)
        return;

    g_Output[i] = g_Textures[i].SampleLevel(float2(0.5f, 0.5f), 0.0f);
}
