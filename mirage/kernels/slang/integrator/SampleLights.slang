// Port of Renderer.cpp's SampleLights(): next-event-estimation over area
// lights with MIS against BSDF sampling, plus sky-probe importance sampling
// when the scene has an HDR environment (sampling/Probe.slang).

import Constants;
import MathExt;
import Random;
import Material;
import Disney;
import SceneBuffers;
import Primitive;
import Trace;
import Sky;
import Probe;
import PunctualLight;

static const float kBsdfSamples = 1.0f;
static const float kProbeSamples = 1.0f;

float3 SampleLights(float3 surfacePos, float3 surfaceNormal, float3 shadingNormal, GpuMaterial surfaceMaterial,
                     float etaI, float etaO, float3 wo, float time, uint numScenePrims, uint numLights, inout Random rand)
{
    float3 sum = float3(0.0f);

    GpuSky sky = g_Sky[0];
    if (sky.probeValid != 0)
    {
        [loop]
        for (int i = 0; i < int(kProbeSamples); ++i)
        {
            float3 skyColor, wi;
            float skyPdf;
            ProbeSample(sky.probeWidth, sky.probeHeight, rand, wi, skyColor, skyPdf);

            float t;
            float3 n;
            int hitPrimIndex;
            float2 shadowUV; // shadow ray - occlusion only, UV/tangent unused
            float3 shadowTangent;
            float3 shadowOrigin = surfacePos + FaceForward(surfaceNormal, wi) * kRayEpsilon;
            if (!TraceWithCutout(shadowOrigin, wi, time, numScenePrims, rand, t, n, hitPrimIndex, shadowUV, shadowTangent))
            {
                float bsdfPdf = BSDFPdf(surfaceMaterial, etaI, etaO, surfacePos, surfaceNormal, wo, wi);
                float3 f = BSDFEval(surfaceMaterial, etaI, etaO, surfacePos, surfaceNormal, wo, wi);

                if (bsdfPdf > 0.0f)
                {
                    float N = kProbeSamples + kBsdfSamples;
                    float cbsdf = kBsdfSamples / N;
                    float csky = kProbeSamples / N;
                    float weight = csky * skyPdf / (cbsdf * bsdfPdf + csky * skyPdf);

                    if (weight > 0.0f)
                        sum += weight * skyColor * f * abs(dot(wi, surfaceNormal)) / skyPdf;
                }
            }
        }

        if (kProbeSamples > 0.0f)
            sum /= kProbeSamples;
    }

    for (uint i = 0; i < numScenePrims; ++i)
    {
        GpuPrimitive lightPrim = g_Primitives[i];
        int numSamples = lightPrim.lightSamples;
        if (numSamples == 0)
            continue;

        float3 L = float3(0.0f);

        for (int s = 0; s < numSamples; ++s)
        {
            float3 lightPos, lightNormal;
            PrimitiveSample(lightPrim, time, rand, lightPos, lightNormal);

            float3 wi = lightPos - surfacePos;
            float dSq = dot(wi, wi);
            wi /= sqrt(dSq);

            float t;
            float3 n;
            int hitPrimIndex;
            float2 shadowUV; // shadow ray - occlusion only, UV/tangent unused
            float3 shadowTangent;
            float3 shadowOrigin = surfacePos + FaceForward(surfaceNormal, wi) * kRayEpsilon;
            if (TraceWithCutout(shadowOrigin, wi, time, numScenePrims, rand, t, n, hitPrimIndex, shadowUV, shadowTangent))
            {
                float tSq = t * t;
                const float kTolerance = 1.e-2f;

                if (abs(t - sqrt(dSq)) <= kTolerance)
                {
                    float nl = abs(dot(lightNormal, wi));
                    if (abs(nl) < 1.e-6f)
                        continue;

                    float lightArea = PrimitiveArea(lightPrim);
                    float lightPdf = ((1.0f / lightArea) * tSq) / nl;

                    float bsdfPdf = BSDFPdf(surfaceMaterial, etaI, etaO, surfacePos, shadingNormal, wo, wi);
                    float3 f = BSDFEval(surfaceMaterial, etaI, etaO, surfacePos, shadingNormal, wo, wi);

                    if (bsdfPdf > 0.0f)
                    {
                        float N = float(lightPrim.lightSamples) + kBsdfSamples;
                        float cbsdf = kBsdfSamples / N;
                        float clight = float(lightPrim.lightSamples) / N;
                        float weight = clight * lightPdf / (cbsdf * bsdfPdf + clight * lightPdf);

                        GpuMaterial hitMaterial = g_Materials[g_Primitives[hitPrimIndex].materialIndex];
                        L += weight * f * hitMaterial.emission * (abs(dot(wi, shadingNormal)) / max(1.e-3f, lightPdf));
                    }
                }
            }
        }

        sum += L * (1.0f / float(numSamples));
    }

    // Punctual (point/directional) lights - see PunctualLight.slang/
    // lights/PunctualLight.h's comment on why this is deliberately NOT
    // MIS-weighted against bsdfPdf like the area-light loop above (a delta
    // light has no surface a BSDF-sampled ray could ever hit).
    for (uint i = 0; i < numLights; ++i)
    {
        GpuLight light = g_Lights[i];

        float3 wi, radiance;
        float dist;
        PunctualLight_Sample(light, surfacePos, rand, wi, radiance, dist);

        float t;
        float3 n;
        int hitPrimIndex;
        float2 shadowUV;
        float3 shadowTangent;
        float3 shadowOrigin = surfacePos + FaceForward(surfaceNormal, wi) * kRayEpsilon;
        bool traced = TraceWithCutout(shadowOrigin, wi, time, numScenePrims, rand, t, n, hitPrimIndex, shadowUV, shadowTangent);

        // Unlike the area-light loop above (whose light IS real geometry
        // the shadow ray is expected to hit), a punctual light has no
        // surface to hit - "unoccluded" here means the shadow ray found
        // nothing before reaching the light's distance.
        const float kTolerance = 1.e-2f;
        if (traced && t < dist - kTolerance)
            continue;

        float3 f = BSDFEval(surfaceMaterial, etaI, etaO, surfacePos, shadingNormal, wo, wi);
        sum += f * radiance * abs(dot(wi, shadingNormal));
    }

    return sum;
}
