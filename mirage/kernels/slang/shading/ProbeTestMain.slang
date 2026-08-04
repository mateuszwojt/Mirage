// M4 validation-only entry point: exercises ProbeSample/ProbeEval/ProbePdf in
// isolation (no scene/BVH/camera/materials involved - only the probe texture +
// CDF/PDF buffers uploaded by VulkanRenderer::UploadScene()). Compared against
// the same Mirage::ProbeSample/ProbeEval/ProbePdf calls on the CPU in
// tools/kernel_validate/KernelValidate.cpp.

import Random;
import Probe;

struct ProbeTestCase
{
    uint seed;
    float3 evalDir;
    float3 pdfDir;
};

struct ProbeTestResult
{
    float3 sampleDir;
    float3 sampleColor;
    float samplePdf;
    float3 evalColor;
    float pdfAtDir;
};

struct ProbeParams
{
    int width;
    int height;
    uint count;
};

uniform StructuredBuffer<ProbeParams> g_Frame;
uniform StructuredBuffer<ProbeTestCase> g_Cases;
uniform RWStructuredBuffer<ProbeTestResult> g_Results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    ProbeParams params = g_Frame[0];
    if (i >= params.count)
        return;

    ProbeTestCase tc = g_Cases[i];
    Random rand = Random(tc.seed);

    ProbeTestResult result;
    ProbeSample(params.width, params.height, rand, result.sampleDir, result.sampleColor, result.samplePdf);
    result.evalColor = ProbeEval(ProbeDirToUV(tc.evalDir));
    result.pdfAtDir = ProbePdf(params.width, params.height, tc.pdfDir);

    g_Results[i] = result;
}
