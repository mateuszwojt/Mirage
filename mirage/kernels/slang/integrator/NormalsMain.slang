// M1 validation-only entry point: mirrors CpuRenderer::Render()'s eNormals mode
// (camera ray generation + Trace(), output hit normal *0.5+0.5 as color, or
// black on a miss). Not the real path integrator - PathTraceMain.slang (M3)
// handles the real ePathTrace mode; RenderMode dispatch between the two stays
// on the host side (VulkanRenderer::Render()), not folded into one kernel.

import Random;
import CameraSampler;
import Trace;
import FrameParams;

uniform StructuredBuffer<GpuFrameParams> g_Frame;
uniform StructuredBuffer<GpuCamera> g_Camera;
uniform RWStructuredBuffer<float4> g_Output;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    GpuFrameParams frame = g_Frame[0];
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;

    uint pixelIndex = pixel.y * frame.width + pixel.x;

    GpuCamera cam = g_Camera[0];
    Random rand = Random(pixelIndex);

    float3 origin, dir;
    CameraSampler_GenerateRay(cam, float(pixel.x), float(pixel.y), rand, origin, dir);

    float t;
    float3 n;
    int primIndex;
    float2 uv; // unused - eNormals mode never shades a hit
    if (Trace(origin, dir, 1.0f, frame.numPrimitives, t, n, primIndex, uv))
    {
        float3 color = n * 0.5f + 0.5f;
        g_Output[pixelIndex] = float4(color, 1.0f);
    }
    else
    {
        g_Output[pixelIndex] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
}
