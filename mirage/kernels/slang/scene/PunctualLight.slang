// Port of lights/PunctualLight.h's PunctualLightSample. See that header's
// comment for why punctual lights get no lightPdf/MIS weight here - a delta
// (or near-delta, radius/angle > 0) light can never be coincidentally hit by
// BSDF sampling, so every sample from this function is meant to be added at
// full NEE weight by the caller (SampleLights.slang).

import MathExt;
import Random;
import Sampler;
import SceneBuffers;

void PunctualLight_Sample(GpuLight light, float3 surfacePos, inout Random rand,
                           out float3 wi, out float3 radiance, out float dist)
{
    if (light.type == 0) // point
    {
        float3 samplePos = light.position;
        if (light.radius > 0.0f)
        {
            float u1, u2;
            Sample2D(rand, u1, u2);
            samplePos += UniformSampleSphere(u1, u2) * light.radius;
        }

        float3 toLight = samplePos - surfacePos;
        float dSq = dot(toLight, toLight);
        dist = sqrt(dSq);
        wi = dist > 0.0f ? toLight / dist : float3(0.0f, 1.0f, 0.0f);

        radiance = light.color * light.intensity / max(dSq, 1.e-6f);
    }
    else // directional
    {
        float3 dir = normalize(light.direction);
        wi = -dir;

        if (light.angle > 0.0f)
        {
            float3 u, v;
            BasisFromVector(wi, u, v);

            float u1, u2;
            Sample2D(rand, u1, u2);
            float2 disc = UniformSampleDisc(u1, u2) * tan(light.angle * 0.5f);
            wi = normalize(wi + u * disc.x + v * disc.y);
        }

        // See PunctualLight.h's identical comment - a large sentinel
        // distance both represents "infinitely far" and keeps the caller's
        // existing occlusion-distance check working unmodified.
        dist = 1.e8f;
        radiance = light.color * light.intensity;
    }
}
