/*
# Copyright Disney Enterprises, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License
# and the following modification to it: Section 6 Trademarks.
# deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the
# trade names, trademarks, service marks, or product names of the
# Licensor and its affiliates, except as required for reproducing
# the content of the NOTICE file.
#
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Adapted to C++ by Miles Macklin 2016

*/

#include "mirage/shaders/Material.h"
#include "mirage/math/Geometric.h"
#include "mirage/core/Sampler.h"

#define USE_UNIFORM_SAMPLING 0
#define USE_SIMPLE_BSDF 0

namespace Mirage
{
	enum BSDFType
	{
		eReflected,
		eTransmitted,
		eSpecular
	};

	CUDA_CALLABLE inline bool Refract(const Vec3 &wi, const Vec3 &n, float eta, Vec3 &wt)
	{
		float cosThetaI = Dot(n, wi);
		float sin2ThetaI = Max(0.0f, float(1.0f - cosThetaI * cosThetaI));
		float sin2ThetaT = eta * eta * sin2ThetaI;

		// total internal reflection
		if (sin2ThetaT >= 1)
			return false;

		float cosThetaT = sqrtf(1.0f - sin2ThetaT);
		wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Vec3(n);
		return true;
	}

	CUDA_CALLABLE inline float SchlickFresnel(float u)
	{
		float m = Clamp(1 - u, 0.0f, 1.0f);
		float m2 = m * m;
		return m2 * m2 * m; // pow(m,5)
	}

	CUDA_CALLABLE inline float GTR1(float NDotH, float a)
	{
		if (a >= 1)
			return kInvPi;
		float a2 = a * a;
		float t = 1 + (a2 - 1) * NDotH * NDotH;
		return (a2 - 1) / (kPi * logf(a2) * t);
	}

	CUDA_CALLABLE inline float GTR2(float NDotH, float a)
	{
		float a2 = a * a;
		float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
		return a2 / (kPi * t * t);
	}

	CUDA_CALLABLE inline float SmithGGX(float NDotv, float alphaG)
	{
		float a = alphaG * alphaG;
		float b = NDotv * NDotv;
		return 1 / (NDotv + sqrtf(a + b - a * b));
	}

	CUDA_CALLABLE inline float Fr(float VDotN, float etaI, float etaT)
	{
		float SinThetaT2 = Sqr(etaI / etaT) * (1.0f - VDotN * VDotN);

		// total internal reflection
		if (SinThetaT2 > 1.0f)
			return 1.0f;

		float LDotN = sqrtf(1.0f - SinThetaT2);

		// todo: reformulate to remove this division
		float eta = etaT / etaI;

		float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
		float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);

		return 0.5f * (Sqr(r1) + Sqr(r2));
	}

// lambert
#if USE_SIMPLE_BSDF

	CUDA_CALLABLE inline float BSDFPdf(const Material &mat, float etaI, float etaO, const Vec3 &P, const Vec3 &n, const Vec3 &V, const Vec3 &L)
	{
		if (Dot(L, n) <= 0.0f)
			return 0.0f;
		else
			return kInv2Pi;
	}

	CUDA_CALLABLE inline void BSDFSample(const Material &mat, float etaI, float etaO, const Vec3 &P, const Vec3 &U, const Vec3 &V, const Vec3 &N, const Vec3 &view, Vec3 &light, float &pdf, BSDFType &type, Random &rand)
	{
		Vec3 d = UniformSampleHemisphere(rand);

		light = U * d.x + V * d.y + N * d.z;
		pdf = kInv2Pi;
		type = eReflected;
	}

	CUDA_CALLABLE inline Vec3 BSDFEval(const Material &mat, float etaI, float etaO, const Vec3 &P, const Vec3 &N, const Vec3 &V, const Vec3 &L)
	{
		return kInvPi * mat.color;
	}

#else

	CUDA_CALLABLE inline float BSDFPdf(const Material &mat, float etaI, float etaO, const Vec3 &P, const Vec3 &n, const Vec3 &V, const Vec3 &L)
	{
#if USE_UNIFORM_SAMPLING

		return kInv2Pi * 0.5f;

#endif

		if (Dot(L, n) <= 0.0f)
		{

			float bsdfPdf = 0.0f;
			float brdfPdf = kInv2Pi * mat.subsurface * 0.5f;

			return Lerp(brdfPdf, bsdfPdf, mat.transmission);
		}
		else
		{
			float F = Fr(Dot(n, V), etaI, etaO);

			const float a = Max(0.001f, mat.roughness);

			const Vec3 half = SafeNormalize(L + V);

			const float cosThetaHalf = Abs(Dot(half, n));
			const float pdfHalf = GTR2(cosThetaHalf, a) * cosThetaHalf;

			// calculate pdf for each method given outgoing light vector
			float pdfSpec = 0.25f * pdfHalf / Max(1.e-6f, Dot(L, half));
			// assert(isfinite(pdfSpec));

			float pdfDiff = Abs(Dot(L, n)) * kInvPi * (1.0f - mat.subsurface);
			// assert(isfinite(pdfDiff));

			float bsdfPdf = pdfSpec * F;
			float brdfPdf = Lerp(pdfDiff, pdfSpec, 0.5f);

			// weight pdfs equally
			return Lerp(brdfPdf, bsdfPdf, mat.transmission);
		}
	}

	// generate an importance sampled BSDF direction
	CUDA_CALLABLE inline void BSDFSample(const Material &mat, float etaI, float etaO, const Vec3 &P, const Vec3 &U, const Vec3 &V, const Vec3 &N, const Vec3 &view, Vec3 &light, float &pdf, BSDFType &type, Random &rand)
	{
		if (rand.Randf() < mat.transmission)
		{
			// sample BSDF
			float F = Fr(Dot(N, view), etaI, etaO);

			// sample reflectance or transmission based on Fresnel term
			if (rand.Randf() < F)
			{
				// sample specular
				float r1, r2;
				Sample2D(rand, r1, r2);

				const float a = Max(0.001f, mat.roughness);
				const float phiHalf = r1 * k2Pi;

				const float cosThetaHalf = sqrtf((1.0f - r2) / (1.0f + (Sqr(a) - 1.0f) * r2));
				const float sinThetaHalf = sqrtf(Max(0.0f, 1.0f - Sqr(cosThetaHalf)));
				const float sinPhiHalf = sinf(phiHalf);
				const float cosPhiHalf = cosf(phiHalf);

				Vec3 half = U * (sinThetaHalf * cosPhiHalf) + V * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;

				// ensure half angle in same hemisphere as incoming light vector
				if (Dot(half, view) <= 0.0f)
					half *= -1.0f;

				type = eReflected;
				light = 2.0f * Dot(view, half) * half - view;
			}
			else
			{
				// sample transmission
				float eta = etaI / etaO;

				// Vec3 h = Normalize(V+light);

				if (Refract(view, N, eta, light))
				{
					type = eSpecular;
					pdf = (1.0f - F) * mat.transmission;
					return;
				}
				else
				{
					// assert(0);
					pdf = 0.0f;
					return;
				}
			}
		}
		else
		{

#if USE_UNIFORM_SAMPLING

			light = UniformSampleSphere(rand.Randf(), rand.Randf());
			pdf = kInv2Pi * 0.5f;

			return;
#else

			// sample brdf
			float r1, r2;
			Sample2D(rand, r1, r2);

			if (rand.Randf() < 0.5f)
			{
				// sample diffuse
				if (rand.Randf() < mat.subsurface)
				{
					const Vec3 d = UniformSampleHemisphere(rand);

					// negate z coordinate to sample inside the surface
					light = U * d.x + V * d.y - N * d.z;
					type = eTransmitted;
				}
				else
				{
					const Vec3 d = CosineSampleHemisphere(r1, r2);

					light = U * d.x + V * d.y + N * d.z;
					type = eReflected;
				}
			}
			else
			{
				// sample specular
				const float a = Max(0.001f, mat.roughness);

				const float phiHalf = r1 * k2Pi;

				const float cosThetaHalf = sqrtf((1.0f - r2) / (1.0f + (Sqr(a) - 1.0f) * r2));
				const float sinThetaHalf = sqrtf(Max(0.0f, 1.0f - Sqr(cosThetaHalf)));
				const float sinPhiHalf = sinf(phiHalf);
				const float cosPhiHalf = cosf(phiHalf);

				Vec3 half = U * (sinThetaHalf * cosPhiHalf) + V * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;

				// ensure half angle in same hemisphere as incoming light vector
				if (Dot(half, view) <= 0.0f)
					half *= -1.0f;

				light = 2.0f * Dot(view, half) * half - view;
				type = eReflected;
			}
#endif
		}

		pdf = BSDFPdf(mat, etaI, etaO, P, N, view, light);
	}

	CUDA_CALLABLE inline Vec3 BSDFEval(const Material &mat, float etaI, float etaO, const Vec3 &P, const Vec3 &N, const Vec3 &V, const Vec3 &L)
	{
		float NDotL = Dot(N, L);
		float NDotV = Dot(N, V);

		Vec3 H = Normalize(L + V);

		float NDotH = Dot(N, H);
		float LDotH = Dot(L, H);

		Vec3 Cdlin = Vec3(mat.color);
		float Cdlum = .3 * Cdlin[0] + .6 * Cdlin[1] + .1 * Cdlin[2]; // luminance approx.

		Vec3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : Vec3(1.0f); // normalize lum. to isolate hue+sat
		Vec3 Cspec0 = Lerp(mat.specular * .08 * Lerp(Vec3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
		// Vec3 Csheen = Lerp(Vec3(1), Ctint, mat.sheenTint);

		Vec3 bsdf = 0.0f;
		Vec3 brdf = 0.0f;

		if (mat.transmission > 0.0f)
		{
			// evaluate BSDF
			if (NDotL <= 0)
			{
				// transmission Fresnel
				float F = Fr(NDotV, etaI, etaO);

				bsdf = mat.transmission * (1.0f - F) / Abs(NDotL) * (1.0f - mat.metallic);
			}
			else
			{
				// specular lobe
				float a = Max(0.001f, mat.roughness);
				float Ds = GTR2(NDotH, a);

				// Fresnel term with the microfacet normal
				float FH = Fr(LDotH, etaI, etaO);

				Vec3 Fs = Lerp(Cspec0, Vec3(1.0f), FH);
				float roughg = a;
				float Gs = SmithGGX(NDotV, roughg) * SmithGGX(NDotL, roughg);

				bsdf = Gs * Fs * Ds;
			}
		}

		if (mat.transmission < 1.0f)
		{
			// evaluate BRDF
			if (NDotL <= 0)
			{
				if (mat.subsurface > 0.0f)
				{
					// take sqrt to account for entry/exit of the ray through the medium
					// this ensures transmitted light corresponds to the diffuse model
					Vec3 s = Vec3(sqrtf(mat.color.x), sqrtf(mat.color.y), sqrtf(mat.color.z));

					float FL = SchlickFresnel(Abs(NDotL)), FV = SchlickFresnel(NDotV);
					float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);

					brdf = kInvPi * s * mat.subsurface * Fd * (1.0f - mat.metallic);
				}
			}
			else
			{
				// specular
				float a = Max(0.001f, mat.roughness);
				float Ds = GTR2(NDotH, a);

				// Fresnel term with the microfacet normal
				float FH = SchlickFresnel(LDotH);

				Vec3 Fs = Lerp(Cspec0, Vec3(1), FH);
				float roughg = a;
				float Gs = SmithGGX(NDotV, roughg) * SmithGGX(NDotL, roughg);

				// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
				// and mix in diffuse retro-reflection based on roughness
				float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
				float Fss90 = LDotH * LDotH * mat.roughness;
				float Fd90 = 0.5 + (Fss90 + Fss90);
				float Fd = Lerp(1.0f, Fd90, FL) * Lerp(1.0f, Fd90, FV);

				// Based on Hanrahan-Krueger BSDF approximation of isotrokPic bssrdf
				// 1.25 scale is used to (roughly) preserve albedo
				// Fss90 used to "flatten" retroreflection based on roughness
				// float Fss = Lerp(1.0f, Fss90, FL) * Lerp(1.0f, Fss90, FV);
				// float ss = 1.25 * (Fss * (1.0f / (NDotL + NDotV) - .5) + .5);

				// clearcoat (ior = 1.5 -> F0 = 0.04)
				float Dr = GTR1(NDotH, Lerp(.1, .001, mat.clearcoatGloss));
				float Fc = Lerp(.04f, 1.0f, FH);
				float Gr = SmithGGX(NDotL, .25) * SmithGGX(NDotV, .25);

				// sheen
				// Vec3 Fsheen = FH * mat.sheen * Csheen;

				// Vec3 out = ((1/kPi) * Lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
				// 	* (1-mat.metallic)*(1.0f-mat.transmission)
				// 	+ Gs*Fs*Ds + .25*mat.clearcoat*Gr*Fr*Dr;

				brdf = kInvPi * Fd * Cdlin * (1.0f - mat.metallic) * (1.0f - mat.subsurface) + Gs * Fs * Ds + mat.clearcoat * Gr * Fc * Dr;
			}
		}

		return Lerp(brdf, bsdf, mat.transmission);
	}
#endif
}
