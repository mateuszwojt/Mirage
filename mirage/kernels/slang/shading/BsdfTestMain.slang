// M2 validation-only entry point: evaluates BSDFEval/BSDFPdf/BSDFSample for a
// batch of independent test cases (no scene/BVH/camera involved - this is a
// unit-test kernel for shading/Disney.slang in isolation). Compared against
// the same Mirage::BSDFEval/BSDFPdf/BSDFSample calls on the CPU in
// tools/kernel_validate/KernelValidate.cpp.

import Random;
import MathExt;
import Material;
import Disney;

struct BsdfTestCase
{
    GpuMaterial material;
    float etaI, etaO;
    float3 n;
    float3 v;
    float3 l;
    uint seed;
};

struct BsdfTestResult
{
    float3 eval;
    float pdf;
    float3 sampleDir;
    float samplePdf;
    int sampleType;
};

struct CountParams
{
    uint count;
};

uniform StructuredBuffer<CountParams> g_Frame;
uniform StructuredBuffer<BsdfTestCase> g_Cases;
uniform RWStructuredBuffer<BsdfTestResult> g_Results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    if (i >= g_Frame[0].count)
        return;

    BsdfTestCase tc = g_Cases[i];
    float3 P = float3(0.0f);

    BsdfTestResult result;
    result.eval = BSDFEval(tc.material, tc.etaI, tc.etaO, P, tc.n, tc.v, tc.l);
    result.pdf = BSDFPdf(tc.material, tc.etaI, tc.etaO, P, tc.n, tc.v, tc.l);

    float3 U, V;
    BasisFromVector(tc.n, U, V);

    Random rand = Random(tc.seed);
    float3 sampleDir;
    float samplePdf;
    int sampleType;
    BSDFSample(tc.material, tc.etaI, tc.etaO, P, U, V, tc.n, tc.v, sampleDir, samplePdf, sampleType, rand);

    result.sampleDir = sampleDir;
    result.samplePdf = samplePdf;
    result.sampleType = sampleType;

    g_Results[i] = result;
}
