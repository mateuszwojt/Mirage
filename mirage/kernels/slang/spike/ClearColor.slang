// M0 validation-only kernel: exercises device creation, dispatch, buffer readback,
// and that the common/ modules import and compile correctly. Not part of the
// integrator - superseded by kernels/slang/integrator/PathTraceMain.slang in M3.

import Random;
import MathExt;

struct FrameParams
{
    uint width;
    uint height;
};

// StructuredBuffer, not ConstantBuffer: a bare top-level ConstantBuffer<T> global
// mixed with another global resource silently fails to deliver its data on this
// Vulkan/MoltenVK + slang-rhi combination (reads back as all-zero) - confirmed via
// the M0 spike's DebugProbe tool. StructuredBuffer<T> uses the same binding path
// as RWStructuredBuffer, which is proven to work.
uniform StructuredBuffer<FrameParams> g_Frame;
uniform RWStructuredBuffer<float4> g_Output;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    FrameParams frame = g_Frame[0];
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= frame.width || pixel.y >= frame.height)
        return;

    uint pixelIndex = pixel.y * frame.width + pixel.x;

    // Exercise Random + MathExt so the common/ module chain is validated, not just
    // the bare dispatch/readback path.
    Random rand = Random(pixelIndex);
    float r = rand.Randf();
    float3 dir = SafeNormalize(float3(float(pixel.x), float(pixel.y), 1.0f));

    g_Output[pixelIndex] = float4(r, dir.x * 0.5f + 0.5f, dir.y * 0.5f + 0.5f, 1.0f);
}
