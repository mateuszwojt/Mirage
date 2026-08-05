// Port of Renderer.cpp's PathTrace(): the main unidirectional path-tracing
// integrator loop (BSDF sampling, absorption/transmission tracking, MIS
// between BSDF and light sampling, including the sky-probe MIS-weighting
// branch on a miss when the scene has an HDR environment).

import Constants;
import MathExt;
import Random;
import Material;
import Disney;
import SceneBuffers;
import Primitive;
import Trace;
import SampleLights;
import Sky;
import Probe;
import NormalMapping;

static const float kIntegratorBsdfSamples = 1.0f;
static const float kIntegratorProbeSamples = 1.0f;

float3 PathTrace(float3 startOrigin, float3 startDir, float time, int maxDepth, uint numScenePrims, inout Random rand,
                  out float outDepth, out float3 outNormal, out int outPrimId)
{
    // Primary-ray AOV capture defaults - overwritten below on a primary-ray
    // hit (i == 0). Left at these sentinels on a primary-ray miss, mirroring
    // the CPU PathTrace()'s same behavior.
    outDepth = 0.0f;
    outNormal = float3(0.0f);
    outPrimId = -1;

    float3 pathThroughput = float3(1.0f, 1.0f, 1.0f);
    float3 totalRadiance = float3(0.0f, 0.0f, 0.0f);

    float3 rayOrigin = startOrigin;
    float3 rayDir = startDir;
    float rayTime = time;
    float rayEta = 1.0f;
    float3 rayAbsorption = float3(0.0f);
    int rayType = kReflected;

    float bsdfPdf = 1.0f;

    for (int i = 0; i < maxDepth; ++i)
    {
        float t;
        float3 n;
        int hitPrimIndex;
        float2 hitUV;
        float3 hitTangent;

        if (TraceWithCutout(rayOrigin, rayDir, rayTime, numScenePrims, rand, t, n, hitPrimIndex, hitUV, hitTangent))
        {
            GpuPrimitive hitPrim = g_Primitives[hitPrimIndex];
            GpuMaterial hitMaterial = g_Materials[hitPrim.materialIndex];

            // Texture overrides - built once here so every subsequent BSDF
            // call this iteration (direct, below, and indirect via
            // SampleLights) sees the same textured values. Mirrors
            // Renderer.cpp's PathTrace() shadedMaterial block exactly.
            if (hitMaterial.albedoTextureIndex >= 0)
                hitMaterial.color = Material_SampleAlbedo(hitMaterial, hitUV);
            if (hitMaterial.roughnessTextureIndex >= 0)
                hitMaterial.roughness = Material_SampleRoughness(hitMaterial, hitUV);
            if (hitMaterial.metallicTextureIndex >= 0)
                hitMaterial.metallic = Material_SampleMetallic(hitMaterial, hitUV);

            // Perturb the shading normal in place - every subsequent use of
            // `n` this iteration (AOV capture, SampleLights, BSDFSample/
            // BSDFEval) sees the bump-mapped normal. Only fires for eMesh
            // hits with both a bound normal map and tangent data (hitTangent
            // is only ever populated in that case).
            if (hitMaterial.normalTextureIndex >= 0)
                n = ApplyNormalMap(n, hitTangent, Material_SampleNormal(hitMaterial, hitUV));

            float outEta;
            float3 outAbsorption;
            if (rayEta == 1.0f)
            {
                outEta = Material_GetIndexOfRefraction(hitMaterial);
                outAbsorption = hitMaterial.absorption;
            }
            else
            {
                outEta = 1.0f;
                outAbsorption = float3(0.0f);
            }

            pathThroughput *= exp(-rayAbsorption * t);

            float3 p = rayOrigin + rayDir * t;

            if (i == 0)
            {
                outDepth = t;
                outNormal = n;
                outPrimId = hitPrim.hydraId;

                totalRadiance += hitMaterial.emission;
            }
            else if (kIntegratorBsdfSamples > 0.0f)
            {
                float lightArea = PrimitiveArea(hitPrim);
                if (lightArea > 0.0f)
                {
                    float lightPdf = ((1.0f / lightArea) * t * t) / clamp(dot(-rayDir, n), 1.e-3f, 1.0f);

                    float N = float(hitPrim.lightSamples) + kIntegratorBsdfSamples;
                    float cbsdf = kIntegratorBsdfSamples / N;
                    float clight = float(hitPrim.lightSamples) / N;
                    float weight = cbsdf * bsdfPdf / (cbsdf * bsdfPdf + clight * lightPdf);

                    if (rayType == kSpecular)
                        weight = 1.0f;

                    totalRadiance += weight * pathThroughput * hitMaterial.emission;
                }
            }

            totalRadiance += pathThroughput * SampleLights(p, n, n, hitMaterial, rayEta, outEta, -rayDir, rayTime, numScenePrims, rand);

            if (hitPrim.lightSamples != 0)
                break;

            float3 u, v;
            BasisFromVector(n, u, v);

            float3 bsdfDir;
            int bsdfType;
            BSDFSample(hitMaterial, rayEta, outEta, p, u, v, n, -rayDir, bsdfDir, bsdfPdf, bsdfType, rand);

            if (bsdfPdf <= 0.0f)
                break;

            float3 f = BSDFEval(hitMaterial, rayEta, outEta, p, n, -rayDir, bsdfDir);

            if (dot(bsdfDir, n) <= 0.0f)
            {
                rayEta = outEta;
                rayType = kTransmitted;
                rayAbsorption = outAbsorption;
            }
            else
            {
                rayType = kReflected;
            }

            pathThroughput *= f * abs(dot(n, bsdfDir)) / bsdfPdf;

            rayType = bsdfType;
            rayDir = bsdfDir;
            rayOrigin = p + FaceForward(n, bsdfDir) * kRayEpsilon;
        }
        else
        {
            float weight = 1.0f;
            GpuSky sky = g_Sky[0];
            if (sky.probeValid != 0 && i > 0 && rayType != kSpecular)
            {
                float skyPdf = ProbePdf(sky.probeWidth, sky.probeHeight, rayDir);

                float N = kIntegratorProbeSamples + kIntegratorBsdfSamples;
                float cbsdf = kIntegratorBsdfSamples / N;
                float csky = kIntegratorProbeSamples / N;

                weight = cbsdf * bsdfPdf / (cbsdf * bsdfPdf + csky * skyPdf);
            }

            totalRadiance += weight * Sky_Eval(rayDir) * pathThroughput;
            break;
        }
    }

    return totalRadiance;
}
