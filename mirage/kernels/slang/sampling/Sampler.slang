// Port of the CUDA_CALLABLE subset of mirage/mirage/core/Sampler.h and the
// distribution-sampling functions from mirage/mirage/math/Geometric.h that
// Disney.slang needs. Host-only pieces (best-candidate/blue-noise precompute)
// stay in C++ - they run once at scene-commit time, never per-ray.

import Random;
import Constants;

// Mirrors Sample1D/Sample2D with USE_RANDOM (the only path compiled into the
// existing CPU/CUDA builds - USE_STRATIFIED/USE_BLUE_NOISE are dead code there).
void Sample1D(inout Random rand, out float u1)
{
    u1 = rand.Randf(0.0f, 1.0f);
}

void Sample2D(inout Random rand, out float u1, out float u2)
{
    u1 = rand.Randf(0.0f, 1.0f);
    u2 = rand.Randf(0.0f, 1.0f);
}

float3 UniformSampleSphere(float u1, float u2)
{
    float z = 1.0f - 2.0f * u1;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * kPi * u2;
    float x = r * cos(phi);
    float y = r * sin(phi);
    return float3(x, y, z);
}

float3 UniformSampleHemisphere(inout Random rand)
{
    float z = rand.Randf(0.0f, 1.0f);
    float w = sqrt(1.0f - z * z);
    float phi = kTwoPi * rand.Randf(0.0f, 1.0f);
    float x = cos(phi) * w;
    float y = sin(phi) * w;
    return float3(x, y, z);
}

float2 UniformSampleDisc(float u1, float u2)
{
    float r = sqrt(u1);
    float theta = kTwoPi * u2;
    return float2(r * cos(theta), r * sin(theta));
}

float3 CosineSampleHemisphere(float u1, float u2)
{
    float2 s = UniformSampleDisc(u1, u2);
    float z = sqrt(max(0.0f, 1.0f - s.x * s.x - s.y * s.y));
    return float3(s.x, s.y, z);
}
