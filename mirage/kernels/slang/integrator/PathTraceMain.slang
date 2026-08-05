// Real per-pixel path-trace entry point (mirrors the existing RenderGpu CUDA
// kernel design - a single megakernel, no wavefront/multi-kernel split; see
// the plan's "Path integrator kernel design" section, which this follows
// directly). Each dispatch = one more accumulated sample per pixel via
// AddSample's box-filter accumulation into g_AccumBuffer; VulkanRenderer
// resolves (divides by weight) at host-side readback time, not here.
//
// Per-pixel RNG is seeded from a hash of (pixelIndex, frameSeed, sampleIndex) -
// intentionally different from the CPU's single shared sequential Random
// stream (independent parallel GPU threads cannot replicate a sequential
// stream). This means per-pixel noise patterns are NOT expected to match the
// CPU renderer's; only the PathTrace()/BSDFSample() functions themselves are
// validated bit-close, via kernel_validate's explicitly-seeded unit tests
// (PathTraceTestMain.slang), not this dispatch.

import Random;
import Sampler;
import CameraSampler;
import FrameParams;
import PathTrace;
import AddSample;
import SceneBuffers;

uniform StructuredBuffer<GpuFrameParams> g_Frame;
uniform StructuredBuffer<GpuCamera> g_Camera;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    GpuFrameParams frame = g_Frame[0];
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;

    uint pixelIndex = pixel.y * frame.width + pixel.x;
    Random rand = Random(pixelIndex ^ (frame.frameSeed * 9781u) ^ (frame.sampleIndex * 6271u));

    GpuCamera cam = g_Camera[0];

    float x, y, timeT;
    Sample1D(rand, timeT); // reuses the same [0,1) draw pattern as Sample2D(x,y) below
    float rx, ry;
    Sample1D(rand, rx);
    Sample1D(rand, ry);
    float time = lerp(cam.shutterStart, cam.shutterEnd, timeT);

    x = float(pixel.x) + rx;
    y = float(pixel.y) + ry;

    float3 origin, dir;
    CameraSampler_GenerateRay(cam, x, y, rand, origin, dir);

    float outDepth;
    float3 outNormal;
    int outPrimId;
    float3 sample = PathTrace(origin, dir, time, frame.maxDepth, frame.numPrimitives, frame.numLights, rand,
                               outDepth, outNormal, outPrimId);

    AddSample(pixelIndex, frame.clampValue, sample);

    // First-hit AOVs - always written (cheap, a few extra stores per pixel);
    // VulkanRenderer.cpp gates the actual host readback on Options::aovMask,
    // not this write.
    g_AovDepth[pixelIndex] = float4(outDepth, 0.0f, 0.0f, 1.0f);
    g_AovNormal[pixelIndex] = float4(outNormal, 1.0f);
    g_AovPrimId[pixelIndex] = float4(float(outPrimId), 0.0f, 0.0f, 1.0f);
}
