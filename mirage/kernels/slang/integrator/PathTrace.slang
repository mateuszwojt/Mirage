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

// Port of Renderer.cpp's MipLevelFromHitDistance - see that function's
// comment for the full rationale (pragmatic footprint-from-hit-distance
// heuristic, not true ray differentials).
float MipLevelFromHitDistance(float hitT, float cameraFovY, uint imageHeight, int texWidth, int texHeight)
{
    if (cameraFovY <= 0.0f || imageHeight == 0u || texWidth <= 0 || texHeight <= 0)
        return 0.0f;

    float pixelAngularSize = 2.0f * tan(cameraFovY * 0.5f) / float(imageHeight);
    float worldFootprint = hitT * pixelAngularSize;
    float texelFootprint = worldFootprint * float(max(texWidth, texHeight));
    return log2(max(1.0f, texelFootprint));
}

// cameraFovY/imageHeight (Tier-2 "mipmapping") mirror Renderer.cpp's
// PathTrace() trailing optional params, except required here (see this
// codebase's "no default Slang params" convention, noted on
// Material_SampleAlbedo et al.) - callers with no camera/mip heuristic
// available (PathTraceTestMain.slang's isolated tests) pass 0.0f/0u,
// which MipLevelFromHitDistance above treats identically to the CPU
// default (mip level always 0).
float3 PathTrace(float3 startOrigin, float3 startDir, float time, int maxDepth, uint numScenePrims, uint numLights,
                  float cameraFovY, uint imageHeight, inout Random rand,
                  out float outDepth, out float3 outNormal, out int outPrimId, out float3 outAlbedo)
{
    // Primary-ray AOV capture defaults - overwritten below on a primary-ray
    // hit (i == 0). Left at these sentinels on a primary-ray miss, mirroring
    // the CPU PathTrace()'s same behavior.
    outDepth = 0.0f;
    outNormal = float3(0.0f);
    outPrimId = -1;
    outAlbedo = float3(0.0f);

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
            // Renderer.cpp's PathTrace() shadedMaterial block exactly,
            // including the mip-level heuristic (primary-ray-only, see
            // MipLevelFromHitDistance's comment - indirect bounces sample
            // mip 0).
            float mipLevel = 0.0f;
            if (i == 0 && hitMaterial.albedoTextureIndex >= 0)
            {
                GpuTextureInfo texInfo = g_TextureInfo[hitMaterial.albedoTextureIndex];
                mipLevel = MipLevelFromHitDistance(t, cameraFovY, imageHeight, texInfo.width, texInfo.height);
            }

            if (hitMaterial.albedoTextureIndex >= 0)
                hitMaterial.color = Material_SampleAlbedo(hitMaterial, hitUV, mipLevel);
            if (hitMaterial.roughnessTextureIndex >= 0)
                hitMaterial.roughness = Material_SampleRoughness(hitMaterial, hitUV, mipLevel);
            if (hitMaterial.metallicTextureIndex >= 0)
                hitMaterial.metallic = Material_SampleMetallic(hitMaterial, hitUV, mipLevel);

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
                // hitMaterial.color is already texture-resolved above
                // (Material_SampleAlbedo), same "direct read of an existing
                // local, no new shading work" reasoning as the CPU mirror.
                outAlbedo = hitMaterial.color;

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

            totalRadiance += pathThroughput * SampleLights(p, n, n, hitMaterial, rayEta, outEta, -rayDir, rayTime, numScenePrims, numLights, rand);

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
