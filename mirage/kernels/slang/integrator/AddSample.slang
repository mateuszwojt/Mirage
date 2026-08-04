// Box-filter, no-atomics accumulation - see the plan's "Concurrency / pixel-
// filter splat" section. Each GPU thread owns exactly one output pixel and
// only ever writes to that pixel (footprint clamped to the sampled texel, not
// filter.width), unlike the CPU/CUDA Gaussian-splat AddSample which scatters
// into a ~3x3 neighborhood. This trades reconstruction-filter quality for
// being correct-by-construction with no cross-thread write hazards on every
// slang-rhi backend. Revisit with atomics (VK_EXT_shader_atomic_float,
// confirmed supported on the M0 dev machine) if Gaussian-quality AA is wanted.

import MathExt;
import SceneBuffers;

void AddSample(uint pixelIndex, float clampValue, float3 sample)
{
    float3 c = ClampLength(sample, clampValue);
    g_AccumBuffer[pixelIndex] += float4(c, 1.0f);
}
