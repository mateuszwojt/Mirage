// Mirrors core/Scene.h's Sky::Eval, including the probe-based HDR environment
// path (sampling/Probe.slang) now that texture upload infrastructure exists.

import Probe;

struct GpuSky
{
    float3 horizon;
    float3 zenith;
    int probeValid;
    int probeWidth;
    int probeHeight;
};

uniform StructuredBuffer<GpuSky> g_Sky;

float3 Sky_Eval(float3 dir)
{
    GpuSky sky = g_Sky[0];
    if (sky.probeValid != 0)
        return ProbeEval(ProbeDirToUV(dir));
    return lerp(sky.horizon, sky.zenith, sqrt(abs(dir.y)));
}
