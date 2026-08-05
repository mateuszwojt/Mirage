// M3 validation-only entry point: runs PathTrace() for a batch of independent,
// explicitly-seeded primary rays (no per-pixel frameSeed/sampleIndex hashing -
// see PathTraceMain.slang's header comment on why that would make CPU/GPU
// comparison statistical rather than exact). Compared against the same
// Mirage::PathTrace() call on the CPU in
// tools/kernel_validate/KernelValidate.cpp.

import Random;
import PathTrace;

struct PathTraceTestCase
{
    float3 origin;
    float3 dir;
    float time;
    int maxDepth;
    uint seed;
    uint numPrimitives;
    uint numLights; // Tier-2 "punctual lights" - 0 for every pre-existing test case
};

struct PathTraceTestResult
{
    float3 radiance;
};

struct CountParams
{
    uint count;
};

uniform StructuredBuffer<CountParams> g_Frame;
uniform StructuredBuffer<PathTraceTestCase> g_Cases;
uniform RWStructuredBuffer<PathTraceTestResult> g_Results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= g_Frame[0].count)
        return;

    PathTraceTestCase tc = g_Cases[i];
    Random rand = Random(tc.seed);

    float unusedDepth;
    float3 unusedNormal;
    int unusedPrimId;

    PathTraceTestResult result;
    result.radiance = PathTrace(tc.origin, tc.dir, tc.time, tc.maxDepth, tc.numPrimitives, tc.numLights, rand,
                                 unusedDepth, unusedNormal, unusedPrimId);

    g_Results[i] = result;
}
