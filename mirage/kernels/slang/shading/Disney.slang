// Port of mirage/mirage/shaders/Disney.h (Disney principled BSDF, adapted to
// C++ by Miles Macklin 2016; original copyright Disney Enterprises, Apache 2.0).
// Mirrors the USE_SIMPLE_BSDF=0 / USE_UNIFORM_SAMPLING=0 path, which is the only
// one actually compiled into the existing CPU/CUDA renderers.

import Constants;
import MathExt;
import Material;
import Sampler;
import Random;

// eReflected=0, eTransmitted=1, eSpecular=2 - matches Disney.h's BSDFType order.
static const int kReflected = 0;
static const int kTransmitted = 1;
static const int kSpecular = 2;

bool Refract(float3 wi, float3 n, float eta, out float3 wt)
{
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    wt = float3(0.0f);
    if (sin2ThetaT >= 1.0f)
        return false;

    float cosThetaT = sqrt(1.0f - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m;
}

float GTR1(float NDotH, float a)
{
    if (a >= 1.0f)
        return kInvPi;
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return (a2 - 1.0f) / (kPi * log(a2) * t);
}

float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return a2 / (kPi * t * t);
}

float SmithGGX(float NDotv, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.0f / (NDotv + sqrt(a + b - a * b));
}

float Fr(float VDotN, float etaI, float etaT)
{
    float sinThetaT2 = (etaI / etaT) * (etaI / etaT) * (1.0f - VDotN * VDotN);
    if (sinThetaT2 > 1.0f)
        return 1.0f;

    float LDotN = sqrt(1.0f - sinThetaT2);
    float eta = etaT / etaI;

    float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
    float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);

    return 0.5f * (r1 * r1 + r2 * r2);
}

float BSDFPdf(GpuMaterial mat, float etaI, float etaO, float3 P, float3 n, float3 V, float3 L)
{
    if (dot(L, n) <= 0.0f)
    {
        float bsdfPdf = 0.0f;
        float brdfPdf = kInvTwoPi * mat.subsurface * 0.5f;
        return lerp(brdfPdf, bsdfPdf, mat.transmission);
    }
    else
    {
        float F = Fr(dot(n, V), etaI, etaO);
        float a = max(0.001f, mat.roughness);
        float3 half = SafeNormalize(L + V);

        float cosThetaHalf = abs(dot(half, n));
        float pdfHalf = GTR2(cosThetaHalf, a) * cosThetaHalf;

        float pdfSpec = 0.25f * pdfHalf / max(1.e-6f, dot(L, half));
        float pdfDiff = abs(dot(L, n)) * kInvPi * (1.0f - mat.subsurface);

        float bsdfPdf = pdfSpec * F;
        float brdfPdf = lerp(pdfDiff, pdfSpec, 0.5f);

        return lerp(brdfPdf, bsdfPdf, mat.transmission);
    }
}

void BSDFSample(GpuMaterial mat, float etaI, float etaO, float3 P, float3 U, float3 V, float3 N, float3 view,
                out float3 light, out float pdf, out int type, inout Random rand)
{
    light = float3(0.0f);
    pdf = 0.0f;
    type = kReflected;

    if (rand.Randf() < mat.transmission)
    {
        float F = Fr(dot(N, view), etaI, etaO);

        if (rand.Randf() < F)
        {
            float r1, r2;
            Sample2D(rand, r1, r2);

            float a = max(0.001f, mat.roughness);
            float phiHalf = r1 * kTwoPi;

            float cosThetaHalf = sqrt((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
            float sinThetaHalf = sqrt(max(0.0f, 1.0f - cosThetaHalf * cosThetaHalf));
            float sinPhiHalf = sin(phiHalf);
            float cosPhiHalf = cos(phiHalf);

            float3 half = U * (sinThetaHalf * cosPhiHalf) + V * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
            if (dot(half, view) <= 0.0f)
                half *= -1.0f;

            type = kReflected;
            light = 2.0f * dot(view, half) * half - view;
        }
        else
        {
            float eta = etaI / etaO;
            float3 refracted;
            if (Refract(view, N, eta, refracted))
            {
                light = refracted;
                type = kSpecular;
                pdf = (1.0f - F) * mat.transmission;
                return;
            }
            else
            {
                pdf = 0.0f;
                return;
            }
        }
    }
    else
    {
        float r1, r2;
        Sample2D(rand, r1, r2);

        if (rand.Randf() < 0.5f)
        {
            if (rand.Randf() < mat.subsurface)
            {
                float3 d = UniformSampleHemisphere(rand);
                light = U * d.x + V * d.y - N * d.z;
                type = kTransmitted;
            }
            else
            {
                float3 d = CosineSampleHemisphere(r1, r2);
                light = U * d.x + V * d.y + N * d.z;
                type = kReflected;
            }
        }
        else
        {
            float a = max(0.001f, mat.roughness);
            float phiHalf = r1 * kTwoPi;

            float cosThetaHalf = sqrt((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
            float sinThetaHalf = sqrt(max(0.0f, 1.0f - cosThetaHalf * cosThetaHalf));
            float sinPhiHalf = sin(phiHalf);
            float cosPhiHalf = cos(phiHalf);

            float3 half = U * (sinThetaHalf * cosPhiHalf) + V * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
            if (dot(half, view) <= 0.0f)
                half *= -1.0f;

            light = 2.0f * dot(view, half) * half - view;
            type = kReflected;
        }
    }

    pdf = BSDFPdf(mat, etaI, etaO, P, N, view, light);
}

float3 BSDFEval(GpuMaterial mat, float etaI, float etaO, float3 P, float3 N, float3 V, float3 L)
{
    float NDotL = dot(N, L);
    float NDotV = dot(N, V);

    float3 H = normalize(L + V);
    float NDotH = dot(N, H);
    float LDotH = dot(L, H);

    float3 Cdlin = mat.color;
    float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z;

    float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : float3(1.0f);
    float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);

    float3 bsdf = float3(0.0f);
    float3 brdf = float3(0.0f);

    if (mat.transmission > 0.0f)
    {
        if (NDotL <= 0.0f)
        {
            float F = Fr(NDotV, etaI, etaO);
            bsdf = mat.transmission * (1.0f - F) / abs(NDotL) * (1.0f - mat.metallic);
        }
        else
        {
            float a = max(0.001f, mat.roughness);
            float Ds = GTR2(NDotH, a);

            float FH = Fr(LDotH, etaI, etaO);
            float3 Fs = lerp(Cspec0, float3(1.0f), FH);
            float roughg = a;
            float Gs = SmithGGX(NDotV, roughg) * SmithGGX(NDotL, roughg);

            bsdf = Gs * Fs * Ds;
        }
    }

    if (mat.transmission < 1.0f)
    {
        if (NDotL <= 0.0f)
        {
            if (mat.subsurface > 0.0f)
            {
                float3 s = float3(sqrt(mat.color.x), sqrt(mat.color.y), sqrt(mat.color.z));

                float FL = SchlickFresnel(abs(NDotL));
                float FV = SchlickFresnel(NDotV);
                float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);

                brdf = kInvPi * s * mat.subsurface * Fd * (1.0f - mat.metallic);
            }
        }
        else
        {
            float a = max(0.001f, mat.roughness);
            float Ds = GTR2(NDotH, a);

            float FH = SchlickFresnel(LDotH);
            float3 Fs = lerp(Cspec0, float3(1.0f), FH);
            float roughg = a;
            float Gs = SmithGGX(NDotV, roughg) * SmithGGX(NDotL, roughg);

            float FL = SchlickFresnel(NDotL);
            float FV = SchlickFresnel(NDotV);
            float Fss90 = LDotH * LDotH * mat.roughness;
            float Fd90 = 0.5f + (Fss90 + Fss90);
            float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

            float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
            float Fc = lerp(0.04f, 1.0f, FH);
            float Gr = SmithGGX(NDotL, 0.25f) * SmithGGX(NDotV, 0.25f);

            brdf = kInvPi * Fd * Cdlin * (1.0f - mat.metallic) * (1.0f - mat.subsurface) + Gs * Fs * Ds + mat.clearcoat * Gr * Fc * Dr;
        }
    }

    return lerp(brdf, bsdf, mat.transmission);
}
