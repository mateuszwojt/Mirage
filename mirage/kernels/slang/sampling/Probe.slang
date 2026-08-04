// Port of core/Probe.h's HDR environment-map importance sampling.
// g_ProbeTexture is used for ProbeEval() (hardware bilinear filtering - the CPU
// version does *no* interpolation/nearest-texel only, so this is an intentional,
// pre-approved small numerical improvement, not a bug - see the plan's
// "Texture / bump-map handling" section). ProbeSample()'s returned color comes
// from g_ProbeData instead (an exact per-texel fetch, matching the CPU's direct
// array indexing bit-for-bit) since that's a discrete CDF-selected texel on both
// sides, not something that should differ.

import Constants;
import Sampler;
import Random;

uniform Sampler2D g_ProbeTexture;
uniform StructuredBuffer<float4> g_ProbeData;
uniform StructuredBuffer<float> g_ProbeCdfX;
uniform StructuredBuffer<float> g_ProbePdfX;
uniform StructuredBuffer<float> g_ProbeCdfY;
uniform StructuredBuffer<float> g_ProbePdfY;

float2 ProbeDirToUV(float3 dir)
{
    float theta = acos(clamp(dir.y, -1.0f, 1.0f));
    float phi = (dir.x == 0.0f && dir.z == 0.0f) ? 0.0f : atan2(dir.z, dir.x);
    float u = (kPi + phi) * kInvPi * 0.5f;
    float v = theta * kInvPi;
    return float2(u, v);
}

float3 ProbeUVToDir(float2 uv)
{
    float theta = uv.y * kPi;
    float phi = uv.x * 2.0f * kPi;

    float x = -sin(theta) * cos(phi);
    float y = cos(theta);
    float z = -sin(theta) * sin(phi);

    return float3(x, y, z);
}

float3 ProbeEval(float2 uv)
{
    return g_ProbeTexture.SampleLevel(uv, 0.0f).rgb;
}

float ProbePdf(int width, int height, float3 d)
{
    float2 uv = ProbeDirToUV(d);

    int col = clamp(int(uv.x * float(width)), 0, width - 1);
    int row = clamp(int(uv.y * float(height)), 0, height - 1);

    float pdf = g_ProbePdfX[row * width + col] * g_ProbePdfY[row];

    float sinTheta = sin(uv.y * kPi);
    if (abs(sinTheta) < 0.0001f)
        pdf = 0.0f;
    else
        pdf *= float(width) * float(height) / (2.0f * kPi * kPi * sinTheta);

    return pdf;
}

// Mirrors Probe.h's integer LowerBound(array, lower, upper, value).
int ProbeLowerBound(StructuredBuffer<float> array, int lo, int hi, float value)
{
    [loop]
    while (lo < hi)
    {
        int mid = lo + (hi - lo) / 2;
        if (array[mid] < value)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

void ProbeSample(int width, int height, inout Random rand, out float3 dir, out float3 color, out float pdf)
{
    float r1, r2;
    Sample2D(rand, r1, r2);

    int row = ProbeLowerBound(g_ProbeCdfY, 0, height, r1);
    int col = ProbeLowerBound(g_ProbeCdfX, row * width, (row + 1) * width, r2) - row * width;

    color = g_ProbeData[row * width + col].rgb;
    pdf = g_ProbePdfX[row * width + col] * g_ProbePdfY[row];

    float u = float(col) / float(width);
    float v = float(row) / float(height);

    float sinTheta = sin(v * kPi);
    if (sinTheta == 0.0f)
        pdf = 0.0f;
    else
        pdf *= float(width) * float(height) / (2.0f * kPi * kPi * sinTheta);

    dir = ProbeUVToDir(float2(u, v));
}
