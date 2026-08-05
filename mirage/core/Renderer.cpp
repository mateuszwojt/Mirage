#include "mirage/core/Renderer.h"
#include "mirage/core/Intersection.h"
#include "mirage/core/Sampler.h"
#include "mirage/shaders/Disney.h"
#include "mirage/shaders/TextureSampling.h"
#include "mirage/shaders/NormalMapping.h"
#include "mirage/utils/Util.h"

#define kBsdfSamples 1.0f
#define kProbeSamples 1.0f
#define kRayEpsilon 0.0001f

// Bounded retry count for stochastic opacity/cutout transparency
// (TraceWithCutout below) - a chain of cutout surfaces can't stall a ray
// indefinitely. This is a caller-side retry loop around Trace(), not a
// BVH-traversal-level skip - QueryBVH/PrimitiveIntersect are untouched by
// this feature (see the plan's "traversal fork" decision).
#define kMaxCutoutRetries 4

// Russian-roulette path termination: bounces below kMinRRBounce always
// continue (keeps early, cheap-to-compute bounces noise-free); from there on
// each bounce survives with a probability based on its throughput, boosted to
// stay unbiased in expectation, instead of always running to the hard
// maxDepth cutoff. kMinRRProb floors the survival probability so throughput
// can never make a path un-terminable.
#define kMinRRBounce 3
#define kMinRRProb 0.05f

#define USE_LIGHT_SAMPLING 1
#define USE_SCENE_BVH 1

namespace Mirage
{
	// Primitive::materialIndex may be unassigned (-1, e.g. a primitive that was
	// never routed through Scene::AddMaterial/FindOrAddMaterial); resolving
	// through this helper rather than Scene::GetMaterial directly preserves the
	// pre-migration behavior where every Primitive always carried a valid,
	// default-constructed Material value.
	inline const Material &ResolveMaterial(const Scene &scene, int materialIndex)
	{
		static const Material kDefaultMaterial;
		const Material *mat = scene.GetMaterial(materialIndex);
		return mat ? *mat : kDefaultMaterial;
	}

	// trace a ray against the scene returning the closest intersection
	// outUV mirrors outNormal: populated only for eMesh hits on a UV-bearing
	// mesh, left at the caller's initial value otherwise (see
	// PrimitiveIntersect's outUV comment in Intersection.h). Defaulted so
	// existing shadow-ray/visibility-only callers (which never shade the hit,
	// only test occlusion - e.g. SampleLights' two Trace() calls) don't need
	// to pass or care about it. outTangent follows the same convention,
	// gated on tangent data instead of UV data (Mesh::HasTangents()).
	inline bool Trace(const Scene &scene, const Ray &ray, float &outT, Vec3 &outNormal, const Primitive **outPrimitive, Vec2 *outUV = nullptr, Vec3 *outTangent = nullptr)
	{

#if USE_SCENE_BVH

		struct Callback
		{
			float minT;
			Vec3 closestNormal;
			Vec2 closestUV;
			Vec3 closestTangent;
			const Primitive *closestPrimitive;

			Ray ray;
			const Scene &scene;

			Callback(const Scene &s, const Ray &r) : minT(REAL_MAX), closestPrimitive(nullptr), ray(r), scene(s)
			{
			}

			void operator()(int index)
			{
				float t;
				Vec3 n, ns;
				Vec2 uv;
				Vec3 tangent;

				const Primitive &primitive = scene.primitives[index];

				if (PrimitiveIntersect(primitive, ray, t, &n, &uv, &tangent))
				{
					if (t < minT && t > 0.0f)
					{
						minT = t;
						closestPrimitive = &primitive;
						closestNormal = n;
						closestUV = uv;
						closestTangent = tangent;
					}
				}
			}
		};

		Callback callback(scene, ray);
		QueryBVH(callback, scene.bvh.nodes, ray.origin, ray.dir);

		outT = callback.minT;
		outNormal = FaceForward(callback.closestNormal, -ray.dir);
		*outPrimitive = callback.closestPrimitive;
		if (outUV)
			*outUV = callback.closestUV;
		if (outTangent)
			*outTangent = callback.closestTangent;

		return callback.closestPrimitive != nullptr;

#else

		// disgard hits closer than this distance to avoid self intersection artifacts
		float minT = REAL_MAX;
		const Primitive *closestPrimitive = nullptr;
		Vec3 closestNormal(0.0f);
		Vec2 closestUV(0.0f);
		Vec3 closestTangent(0.0f);

		for (Scene::PrimitiveArray::const_iterator iter = scene.primitives.begin(), end = scene.primitives.end(); iter != end; ++iter)
		{
			float t;
			Vec3 n, ns;
			Vec2 uv;
			Vec3 tangent;

			const Primitive &primitive = *iter;

			if (PrimitiveIntersect(primitive, ray, t, &n, &uv, &tangent))
			{
				if (t < minT && t > 0.0f)
				{
					minT = t;
					closestPrimitive = &primitive;
					closestNormal = n;
					closestUV = uv;
					closestTangent = tangent;
				}
			}
		}

		outT = minT;
		outNormal = FaceForward(closestNormal, -ray.dir);
		*outPrimitive = closestPrimitive;
		if (outUV)
			*outUV = closestUV;
		if (outTangent)
			*outTangent = closestTangent;

		return closestPrimitive != nullptr;

#endif
	}

	// Traces a ray, transparently skipping past any surface that fails a
	// stochastic opacity test (Material::opacity/opacityTextureIndex) -
	// "cutout" transparency, matching path-tracer convention (accept/reject
	// against a uniform draw, not alpha blending - a surface is either
	// fully there or fully not, per sample). On a reject, the ray continues
	// from just past the rejected hit point, without consuming
	// caller-visible bounce/depth budget. Same signature as Trace() plus
	// `rand`, so every call site swaps in directly. Bounded by
	// kMaxCutoutRetries; if that budget runs out, the last-tested hit is
	// reported as-is (treated as opaque) rather than retried forever.
	inline bool TraceWithCutout(const Scene &scene, Ray ray, float &outT, Vec3 &outNormal, const Primitive **outPrimitive, Random &rand, Vec2 *outUV = nullptr, Vec3 *outTangent = nullptr)
	{
		// Opacity texture sampling needs a UV regardless of whether the
		// caller itself wants one back (most callers here are shadow rays,
		// which pass outUV == nullptr since they only care about occlusion).
		Vec2 uvStorage(0.0f);
		Vec2 *uv = outUV ? outUV : &uvStorage;

		// outT must end up measured from the ORIGINAL ray origin the caller
		// passed in, not from whatever retry-shifted origin the final
		// accepted Trace() call used - SampleLights compares outT against
		// an independently-computed distance-to-light from that same
		// original origin (its "did this shadow ray actually reach the
		// light" check), so without this running total a retried ray would
		// report a too-short distance and wrongly treat a fully-visible
		// light as occluded by whatever it cutout-passed-through en route.
		float accumulatedT = 0.0f;

		for (int i = 0; i < kMaxCutoutRetries; ++i)
		{
			float localT;
			if (!Trace(scene, ray, localT, outNormal, outPrimitive, uv, outTangent))
				return false;

			const Material &mat = ResolveMaterial(scene, (*outPrimitive)->materialIndex);
			float opacity = mat.opacity;
			if (mat.opacityTextureIndex >= 0)
			{
				const Texture *tex = scene.GetTexture(mat.opacityTextureIndex);
				if (tex)
					opacity = SampleTextureAlpha(*tex, *uv);
			}

			outT = accumulatedT + localT;

			if (opacity >= 1.0f || rand.Randf() < opacity)
				return true;

			Vec3 p = ray.origin + ray.dir * localT;
			accumulatedT = outT + kRayEpsilon;
			ray = Ray(p + ray.dir * kRayEpsilon, ray.dir, ray.time);
		}

		return true; // retry budget exhausted - report the last hit as-is
	}

	// Takes the already-resolved (and, once M7's texture overrides are applied
	// by the caller, already-textured) surface Material directly rather than a
	// Primitive to derive it from - matches the GPU's SampleLights.slang, which
	// likewise takes `GpuMaterial surfaceMaterial` by value rather than deriving
	// it from a primitive/material-buffer lookup internally.
	inline Vec3 SampleLights(const Scene &scene, const Material &surfaceMaterial, float etaI, float etaO, const Vec3 &surfacePos, const Vec3 &surfaceNormal, const Vec3 &shadingNormal, const Vec3 &wo, float time, Random &rand)
	{
		Vec3 sum(0.0f);

		if (scene.sky.probe.valid)
		{
			for (int i = 0; i < kProbeSamples; ++i)
			{

				Vec3 skyColor;
				float skyPdf;
				Vec3 wi;

				ProbeSample(scene.sky.probe, wi, skyColor, skyPdf, rand);

				// check if occluded
				float t;
				Vec3 n;
				const Primitive *hit;
				if (TraceWithCutout(scene, Ray(surfacePos + FaceForward(surfaceNormal, wi) * kRayEpsilon, wi, time), t, n, &hit, rand) == false)
				{
					float bsdfPdf = BSDFPdf(surfaceMaterial, etaI, etaO, surfacePos, surfaceNormal, wo, wi);
					Vec3 f = BSDFEval(surfaceMaterial, etaI, etaO, surfacePos, surfaceNormal, wo, wi);

					if (bsdfPdf > 0.0f)
					{
						int N = kProbeSamples + kBsdfSamples;
						float cbsdf = kBsdfSamples / N;
						float csky = float(kProbeSamples) / N;
						float weight = csky * skyPdf / (cbsdf * bsdfPdf + csky * skyPdf);

						if (weight > 0.0f)
							sum += weight * skyColor * f * Abs(Dot(wi, surfaceNormal)) / skyPdf;
					}
				}
			}

			if (kProbeSamples > 0)
				sum /= float(kProbeSamples);
		}

		for (int i = 0; i < scene.primitives.size(); ++i)
		{
			// assume all lights are area lights for now
			const Primitive &lightPrimitive = scene.primitives[i];

			Vec3 L(0.0f);

			int numSamples = lightPrimitive.lightSamples;

			if (numSamples == 0)
				continue;

			for (int s = 0; s < numSamples; ++s)
			{
				// sample light source
				Vec3 lightPos;
				Vec3 lightNormal;

				PrimitiveSample(lightPrimitive, time, lightPos, lightNormal, rand);

				Vec3 wi = lightPos - surfacePos;

				float dSq = LengthSq(wi);
				wi /= sqrtf(dSq);

				// check visibility
				float t;
				Vec3 n;
				const Primitive *hit;
				if (TraceWithCutout(scene, Ray(surfacePos + FaceForward(surfaceNormal, wi) * kRayEpsilon, wi, time), t, n, &hit, rand))
				{
					float tSq = t * t;

					// if our next hit was further than distance to light then accept
					// sample, this works for portal sampling where you have a large light
					// that you sample through a small window
					const float kTolerance = 1.e-2f;

					if (fabsf(t - sqrtf(dSq)) <= kTolerance)
					{
						const float nl = Abs(Dot(lightNormal, wi));

						// for glancing rays, note we use abs to include cases
						// where light surface is backfacing, e.g.: inside the weak furnace
						if (Abs(nl) < 1.e-6f)
							continue;

						// light pdf with respect to area and convert to pdf with respect to solid angle
						float lightArea = PrimitiveArea(lightPrimitive);
						float lightPdf = ((1.0f / lightArea) * tSq) / nl;

						// bsdf pdf for light's direction
						float bsdfPdf = BSDFPdf(surfaceMaterial, etaI, etaO, surfacePos, shadingNormal, wo, wi);
						Vec3 f = BSDFEval(surfaceMaterial, etaI, etaO, surfacePos, shadingNormal, wo, wi);

						// this branch is only necessary to exclude specular paths from light sampling
						// todo: make BSDFEval always return zero for pure specular paths and roll specular eval into BSDFSample()
						if (bsdfPdf > 0.0f)
						{
							// calculate relative weighting of the light and bsdf sampling
							int N = lightPrimitive.lightSamples + kBsdfSamples;
							float cbsdf = kBsdfSamples / N;
							float clight = float(lightPrimitive.lightSamples) / N;
							float weight = clight * lightPdf / (cbsdf * bsdfPdf + clight * lightPdf);

							L += weight * f * ResolveMaterial(scene, hit->materialIndex).emission * (Abs(Dot(wi, shadingNormal)) / Max(1.e-3f, lightPdf));
						}
					}
				}
			}

			sum += L * (1.0f / numSamples);
		}

		// Punctual (point/directional) lights: delta-distribution sources,
		// not part of scene.primitives - see PunctualLight.h. Deliberately
		// NOT MIS-weighted against bsdfPdf like the area-light loop above -
		// a delta light has no surface for a BSDF-sampled ray to ever
		// coincidentally hit, so the balance-heuristic weight degenerates to
		// exactly 1.0 for every sample here; applying the area-light
		// formula (which divides by a lightPdf that doesn't exist for a
		// delta distribution) would be wrong, not just redundant.
		for (const PunctualLight &light : scene.lights)
		{
			Vec3 wi;
			Vec3 radiance;
			float dist;
			PunctualLightSample(light, surfacePos, rand, wi, radiance, dist);

			float t;
			Vec3 n;
			const Primitive *hit;
			bool traced = TraceWithCutout(scene, Ray(surfacePos + FaceForward(surfaceNormal, wi) * kRayEpsilon, wi, time), t, n, &hit, rand);

			// Unlike the area-light loop above (whose light IS real
			// geometry the shadow ray is expected to hit), a punctual light
			// has no surface to hit - "unoccluded" here means the shadow
			// ray found nothing before reaching the light's distance.
			const float kTolerance = 1.e-2f;
			if (traced && t < dist - kTolerance)
				continue;

			Vec3 f = BSDFEval(surfaceMaterial, etaI, etaO, surfacePos, shadingNormal, wo, wi);
			sum += f * radiance * Abs(Dot(wi, shadingNormal));
		}

		return sum;
	}

	// Approximate mip level from a primary-ray hit distance - a pragmatic
	// stand-in for true ray differentials/cone tracing (not implemented;
	// this is a known-approximate placeholder, not a claim of physical
	// accuracy). Estimates a pixel's angular size from the camera's vertical
	// FOV and image height, scales by hit distance to get an approximate
	// world-space texel footprint, and converts that to a mip level via
	// log2 against the texture's largest dimension - the same basic
	// "footprint in texels -> log2" idea every mip-selection scheme uses,
	// just without the anisotropic/ray-differential precision. Deliberately
	// only ever called for i==0 (primary-ray) hits in PathTrace() below -
	// indirect bounces use mip 0 (roughness-based footprint growth is out of
	// scope for this milestone).
	inline float MipLevelFromHitDistance(float hitT, float cameraFovY, int imageHeight, int texWidth, int texHeight, float mipBiasScale = 1.0f)
	{
		if (cameraFovY <= 0.0f || imageHeight <= 0 || texWidth <= 0 || texHeight <= 0)
			return 0.0f;

		float pixelAngularSize = 2.0f * tanf(cameraFovY * 0.5f) / float(imageHeight);
		float worldFootprint = hitT * pixelAngularSize;
		float texelFootprint = worldFootprint * mipBiasScale * float(Max(texWidth, texHeight));
		return log2f(Max(1.0f, texelFootprint));
	}

	// reference, no light sampling, uniform hemisphere sampling. cameraFovY/
	// imageHeight (Tier-2 "mipmapping") are optional - 0 (the default) means
	// "no mip heuristic available, always sample mip level 0," which is
	// every pre-existing call site's exact prior behavior (RunPathTraceValidation's
	// direct PathTrace() calls in kernel_validate, in particular, are not
	// expected to change).
	Vec3 PathTrace(const Scene &scene, const Vec3 &startOrigin, const Vec3 &startDir, float time, int maxDepth, Random &rand,
				   float *outDepth = nullptr, Vec3 *outNormal = nullptr, const Primitive **outHitPrim = nullptr,
				   float cameraFovY = 0.0f, int imageHeight = 0)
	{
		// Primary-ray AOV capture defaults - overwritten below on a primary-ray
		// hit (i == 0). Left at these sentinels on a primary-ray miss, so a
		// background pixel reports depth 0 / normal 0 / primId -1 rather than
		// stale or garbage data.
		if (outDepth) *outDepth = 0.0f;
		if (outNormal) *outNormal = Vec3(0.0f);
		if (outHitPrim) *outHitPrim = nullptr;

		// path throughput
		Vec3 pathThroughput(1.0f, 1.0f, 1.0f);
		// accumulated radiance
		Vec3 totalRadiance(0.0f, 0.0f, 0.0f);

		Vec3 rayOrigin = startOrigin;
		Vec3 rayDir = startDir;
		float rayTime = time;
		float rayEta = 1.0f;
		Vec3 rayAbsorption = 0.0f;
		BSDFType rayType = eReflected;

		float t = 0.0f;
		Vec3 n;
		Vec2 hitUV(0.0f);
		Vec3 hitTangent(0.0f);
		const Primitive *hit;

		float bsdfPdf = 1.0f;

		for (int i = 0; i < maxDepth; ++i)
		{
			// find closest hit
			if (TraceWithCutout(scene, Ray(rayOrigin, rayDir, rayTime), t, n, &hit, rand, &hitUV, &hitTangent))
			{
				// Shaded material: a per-hit copy of the resolved Material with
				// color/roughness/metallic overridden from sampled textures
				// where bound - built once here so every subsequent BSDF call
				// this iteration (direct, via BSDFSample/BSDFEval below, and
				// indirect, via SampleLights) sees the same textured values.
				// hitUV is only meaningful for eMesh hits on a UV-bearing mesh
				// (see Trace()/PrimitiveIntersect()'s outUV comments) - a
				// texture-indexed material on a primitive with no UV data just
				// samples texel (0,0) uniformly, which is safe, not a crash.
				Material shadedMaterial = ResolveMaterial(scene, hit->materialIndex);

				// See MipLevelFromHitDistance's comment - only computed (and
				// non-zero) for the primary ray; indirect bounces sample mip
				// 0, same as before this feature existed.
				float mipLevel = 0.0f;
				if (i == 0 && shadedMaterial.albedoTextureIndex >= 0)
				{
					const Texture *tex = scene.GetTexture(shadedMaterial.albedoTextureIndex);
					if (tex)
						mipLevel = MipLevelFromHitDistance(t, cameraFovY, imageHeight, tex->width, tex->height);
				}

				if (shadedMaterial.albedoTextureIndex >= 0)
				{
					const Texture *tex = scene.GetTexture(shadedMaterial.albedoTextureIndex);
					if (tex)
						shadedMaterial.color = SampleTextureRGB(*tex, hitUV, mipLevel);
				}
				if (shadedMaterial.roughnessTextureIndex >= 0)
				{
					const Texture *tex = scene.GetTexture(shadedMaterial.roughnessTextureIndex);
					if (tex)
						shadedMaterial.roughness = SampleTextureR(*tex, hitUV, mipLevel);
				}
				if (shadedMaterial.metallicTextureIndex >= 0)
				{
					const Texture *tex = scene.GetTexture(shadedMaterial.metallicTextureIndex);
					if (tex)
						shadedMaterial.metallic = SampleTextureR(*tex, hitUV, mipLevel);
				}
				const Material &hitMaterial = shadedMaterial;

				// Perturb the shading normal in place - every subsequent use
				// of `n` this iteration (AOV capture, SampleLights,
				// BSDFSample/BSDFEval) sees the bump-mapped normal, same
				// "build once, reuse everywhere" reasoning as shadedMaterial
				// above. Only fires for eMesh hits with both a bound normal
				// map and tangent data (hitTangent is only ever populated in
				// that case - see PrimitiveIntersect's outTangent comment).
				if (hitMaterial.normalTextureIndex >= 0)
				{
					const Texture *tex = scene.GetTexture(hitMaterial.normalTextureIndex);
					if (tex)
						n = ApplyNormalMap(n, hitTangent, SampleTextureRGB(*tex, hitUV));
				}

				float outEta;
				Vec3 outAbsorption;

				// index of refraction for transmission, 1.0 corresponds to air
				if (rayEta == 1.0f)
				{
					outEta = hitMaterial.GetIndexOfRefraction();
					outAbsorption = Vec3(hitMaterial.absorption);
				}
				else
				{
					// returning to free space
					outEta = 1.0f;
					outAbsorption = 0.0f;
				}

				// update throughput based on absorption through the medium
				pathThroughput *= Exp(-rayAbsorption * t);

				// calculate new ray position
				const Vec3 p = rayOrigin + rayDir * t;

				if (i == 0)
				{
					// First-hit AOV capture - t/n/hit are the primary ray's
					// hit distance, shading normal, and hit primitive, needed
					// for the depth/normal/primId AOVs regardless of which
					// light-sampling code path below actually runs.
					if (outDepth) *outDepth = t;
					if (outNormal) *outNormal = n;
					if (outHitPrim) *outHitPrim = hit;
				}

#if USE_LIGHT_SAMPLING

				if (i == 0)
				{
					// first trace is our only chance to add contribution from directly visible light sources
					totalRadiance += hitMaterial.emission;
				}
				else if (kBsdfSamples > 0)
				{
					// area pdf that this dir was already included by the light sampling from previous step
					float lightArea = PrimitiveArea(*hit);

					if (lightArea > 0.0f)
					{
						// convert to pdf with respect to solid angle
						float lightPdf = ((1.0f / lightArea) * t * t) / Clamp(Dot(-rayDir, n), 1.e-3f, 1.0f);

						// calculate weight for bsdf sampling
						int N = hit->lightSamples + kBsdfSamples;
						float cbsdf = kBsdfSamples / N;
						float clight = float(hit->lightSamples) / N;
						float weight = cbsdf * bsdfPdf / (cbsdf * bsdfPdf + clight * lightPdf);

						// specular paths have zero chance of being included by direct light sampling (zero pdf)
						if (rayType == eSpecular)
							weight = 1.0f;

						// pathThroughput already includes the bsdf pdf
						totalRadiance += weight * pathThroughput * hitMaterial.emission;
					}
				}

				// integrate direct light over hemisphere
				totalRadiance += pathThroughput * SampleLights(scene, hitMaterial, rayEta, outEta, p, n, n, -rayDir, rayTime, rand);
#else

				// include emission from the new primitive
				totalRadiance += pathThroughput * hitMaterial.emission;

#endif

				// terminate ray if we hit a light source
				if (hit->lightSamples)
					break;

				// integrate indirect light by sampling BRDF
				Vec3 u, v;
				BasisFromVector(n, &u, &v);

				Vec3 bsdfDir;
				BSDFType bsdfType;
				BSDFSample(hitMaterial, rayEta, outEta, p, u, v, n, -rayDir, bsdfDir, bsdfPdf, bsdfType, rand);

				if (bsdfPdf <= 0.0f)
					break;

				// reflectance
				Vec3 f = BSDFEval(hitMaterial, rayEta, outEta, p, n, -rayDir, bsdfDir);

				// update ray medium if we are transmitting through the material
				if (Dot(bsdfDir, n) <= 0.0f)
				{
					rayEta = outEta;
					rayType = eTransmitted;
					rayAbsorption = outAbsorption;
				}
				else
				{
					rayType = eReflected;
				}

				// update throughput with primitive reflectance
				pathThroughput *= f * Abs(Dot(n, bsdfDir)) / bsdfPdf;

				// Russian roulette: stochastically terminate low-throughput
				// paths past a minimum bounce count instead of always running
				// to maxDepth. Surviving paths get their throughput boosted
				// by 1/continueProb so this stays unbiased in expectation -
				// glass/deep-GI paths converge instead of being flatly
				// truncated at the hard depth cutoff.
				if (i >= kMinRRBounce)
				{
					float continueProb = Clamp(Max(Max(pathThroughput.x, pathThroughput.y), pathThroughput.z), kMinRRProb, 1.0f);
					if (rand.Randf() > continueProb)
						break;
					pathThroughput /= continueProb;
				}

				// update path direction
				rayType = bsdfType;
				rayDir = bsdfDir;
				rayOrigin = p + FaceForward(n, bsdfDir) * kRayEpsilon;
			}
			else
			{
				// hit nothing, sample sky dome and terminate
				float weight = 1.0f;

				if (scene.sky.probe.valid && i > 0 && rayType != eSpecular)
				{
					// probability that this dir was already sampled by probe sampling
					float skyPdf = ProbePdf(scene.sky.probe, rayDir);

					int N = kProbeSamples + kBsdfSamples;
					float cbsdf = kBsdfSamples / N;
					float csky = float(kProbeSamples) / N;

					weight = cbsdf * bsdfPdf / (cbsdf * bsdfPdf + csky * skyPdf);
				}

				totalRadiance += weight * scene.sky.Eval(rayDir) * pathThroughput;
				break;
			}
		}

		return totalRadiance;
	}

	struct CpuRenderer : public Renderer
	{
		CpuRenderer(const Scene *s) : scene(s)
		{
		}

		const Scene *scene;

		void AddSample(Color *output, int width, int height, float rasterX, float rasterY, float clamp, const Filter &filter, const Vec3 &sample)
		{
			switch (filter.type)
			{
			case eFilterBox:
			{
				int startX = Max(0, int(rasterX - filter.width));
				int startY = Max(0, int(rasterY - filter.width));
				int endX = Min(int(rasterX + filter.width), width - 1);
				int endY = Min(int(rasterY + filter.width), height - 1);

				Vec3 c = ClampLength(sample, clamp);

				for (int x = startX; x <= endX; ++x)
				{
					for (int y = startY; y <= endY; ++y)
					{
						output[y * width + x] += Color(c, 1.0f);
					}
				}

				break;
			}
			case eFilterGaussian:
			{
				int startX = Max(0, int(rasterX - filter.width));
				int startY = Max(0, int(rasterY - filter.width));
				int endX = Min(int(rasterX + filter.width), width - 1);
				int endY = Min(int(rasterY + filter.width), height - 1);

				Vec3 c = ClampLength(sample, clamp);

				for (int x = startX; x <= endX; ++x)
				{
					for (int y = startY; y <= endY; ++y)
					{
						float w = filter.Eval(x - rasterX, y - rasterY);

						output[y * width + x] += Color(c * w, w);
					}
				}
				break;
			}
			};
		}

		void Render(const Camera &camera, const Options &options, Color *output, AovBuffers *aovs = nullptr)
		{
			// create a sampler for the camera
			CameraSampler sampler(
				Transform(camera.position, camera.rotation),
				camera.EffectiveFov(),
				0.001f,
				1.0f,
				camera.EffectiveApertureDiameter(),
				camera.focalPoint,
				options.enableDOF,
				options.width,
				options.height);

			for (int k = 0; k < options.maxSamples; ++k)
			{
				for (int j = 0; j < options.height; ++j)
				{

#pragma omp parallel for schedule(dynamic, 1)

					for (int i = 0; i < options.width; ++i)
					{
						Vec3 origin;
						Vec3 dir;

						// generate a ray
						switch (options.mode)
						{
						case ePathTrace:
						{
							// Per-pixel, per-sample RNG stream, seeded from
							// (i, j, k) rather than sharing one Random across
							// every pixel/thread. A single shared instance
							// would be mutated concurrently by every OpenMP
							// thread with no synchronization (a data race,
							// previously masked by only ever taking 1 sample),
							// and would also correlate samples across pixels
							// instead of decorrelating them - same hash-seed
							// style as the GPU path's per-frame seed (see
							// VulkanRenderer.cpp's frameSeed).
							uint32_t seed = (uint32_t)(j * options.width + i) * 9781u + (uint32_t)k * 6271u + 1u;
							Random rand((int)seed);

							float x, y, t;

							Sample2D(rand, x, y);
							Sample1D(rand, t);

							float time = Lerp(camera.shutterStart, camera.shutterEnd, t);

							x += i;
							y += j;

							sampler.GenerateRay(x, y, origin, dir);

							float hitDepth = 0.0f;
							Vec3 hitNormal;
							const Primitive *hitPrim = nullptr;
							bool wantAovs = (aovs != nullptr) && (options.aovMask != 0) && (k == 0);

							Vec3 sample = PathTrace(*scene, origin, dir, time, options.maxDepth, rand,
													 wantAovs ? &hitDepth : nullptr,
													 wantAovs ? &hitNormal : nullptr,
													 wantAovs ? &hitPrim : nullptr,
													 camera.EffectiveFov(), options.height);

							AddSample(output, options.width, options.height, x, y, options.clamp, options.filter, sample);

							if (wantAovs)
							{
								// Direct per-pixel write (own index, no filter
								// splat) - unlike AddSample's antialiasing
								// scatter, depth/normal/primId are discrete
								// single-valued snapshots that must not blur
								// across neighboring pixels. Only captured on
								// the first sample (k == 0) so AA jitter across
								// samples can't perturb which primitive/depth
								// gets reported near silhouette edges.
								int idx = j * options.width + i;
								if ((options.aovMask & kAovDepth) && aovs->depth)
									aovs->depth[idx] = Color(hitDepth, 0.0f, 0.0f, 1.0f);
								if ((options.aovMask & kAovNormal) && aovs->normal)
									aovs->normal[idx] = Color(hitNormal.x, hitNormal.y, hitNormal.z, 1.0f);
								if ((options.aovMask & kAovPrimId) && aovs->primId)
									aovs->primId[idx] = Color(hitPrim ? (float)hitPrim->hydraId : -1.0f, 0.0f, 0.0f, 1.0f);
							}

							break;
						}
						case eNormals:
						{
							const float x = i; // + 0.5f;
							const float y = j; // + 0.5f;

							sampler.GenerateRay(x, y, origin, dir);

							const Primitive *p;
							float t;
							Vec3 n;

							if (Trace(*scene, Ray(origin, dir, 1.0f), t, n, &p))
							{
								n = n * 0.5f + 0.5f;
								output[j * options.width + i] = Color(n.x, n.y, n.z, 1.0f);
							}
							else
							{
								output[j * options.width + i] = Color(0.0f);
							}
							break;
						}
						case eComplexity:
						{
							break;
						}
						}
					}
				}
			}

			// Resolve: divide each pixel's accumulated color by its
			// accumulated filter weight. AddSample() only ever splats
			// Color(c*w, w) - color and filter weight - so without this the
			// reconstruction filter is unnormalized and pixels near a filter
			// footprint's edge (where accumulated weight < the center-tap
			// weight) come out too dim. Mirrors the resolve step the GPU path
			// does host-side at readback time (VulkanRenderer.cpp's
			// RunPathTrace, dividing acc.rgb by acc.w). Safe for eNormals too:
			// it always writes w == 1 on a hit / w == 0 on a miss directly, so
			// this is a no-op there beyond the guarded miss case. Keeps the
			// raw accumulated weight in alpha (not collapsed to 1) to match
			// the GPU path's output convention.
			for (int idx = 0; idx < options.width * options.height; ++idx)
			{
				float w = output[idx].w;
				float invW = w > 0.0f ? 1.0f / w : 0.0f;
				output[idx] = Color(output[idx].x * invW, output[idx].y * invW, output[idx].z * invW, w);
			}
		}
	};

	Renderer *CreateCpuRenderer(const Scene *s)
	{
		return new CpuRenderer(s);
	}

	struct NullRenderer : public Renderer
	{
		virtual ~NullRenderer() {}

		virtual void Init(int width, int height) {}
		virtual void Render(const Camera &c, const Options &options, Color *output, AovBuffers *aovs = nullptr)
		{
			memset(output, 0, options.width * options.height * sizeof(Color));
		};
	};

	Renderer *CreateNullRenderer(const Scene *s)
	{
		return new NullRenderer();
	}
}
