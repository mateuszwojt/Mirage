#pragma once

#include "mirage/math/Vec3.h"
#include "mirage/core/Sampler.h"
#include "mirage/math/Geometric.h"

// Point/directional lights: delta-distribution punctual lights, deliberately
// kept outside the Primitive/BVH system (see Primitive.h's Type enum
// comment) - they have no surface area, are never camera-visible, and can
// never be hit by BSDF sampling, so they need neither intersection geometry
// nor the area-light MIS machinery every Primitive-based light uses. Rect/
// disk area lights are Primitive::Type extensions instead - see
// Intersection.h's eRect/eDisk cases.

namespace Mirage
{
	enum class PunctualLightType
	{
		ePoint,
		eDirectional
	};

	struct PunctualLight
	{
		PunctualLightType type = PunctualLightType::ePoint;

		// ePoint only: world-space position.
		Vec3 position = Vec3(0.0f);

		// eDirectional only: the direction the light travels (i.e. "sun
		// direction", pointing FROM the light TOWARD the scene) - the
		// direction TO the light for shading purposes is -direction.
		Vec3 direction = Vec3(0.0f, -1.0f, 0.0f);

		Vec3 color = Vec3(1.0f);
		float intensity = 1.0f;

		// ePoint only: soft-shadow sample radius (world units) - 0 = true
		// point delta light, sampled with zero variance (every sample
		// returns the exact same direction). > 0 jitters the sampled
		// position on a small sphere for a soft-shadow approximation, still
		// evaluated as a single NEE sample with no BSDF-side MIS (see
		// PunctualLightSample's comment below).
		float radius = 0.0f;

		// eDirectional only: angular diameter in radians (e.g. the sun's
		// ~0.53 degrees / 0.0093 radians) - 0 = true directional delta light
		// (perfectly parallel rays, zero variance). > 0 jitters the sampled
		// direction within a cone of this angular diameter for a soft-sun
		// approximation.
		float angle = 0.0f;
	};

	// Samples a direction/radiance contribution from `light` as seen from
	// `surfacePos`, mirroring PrimitiveSample's shape (position/normal out
	// params) but for a shape-less delta (or near-delta, if radius/angle > 0)
	// light instead of a Primitive. Unlike PrimitiveArea-based lights,
	// there's no light pdf here to divide by - a delta light can only ever be
	// found by explicit NEE sampling (never coincidentally hit by BSDF
	// sampling, since it has no surface to intersect), so the balance-
	// heuristic MIS weight every other SampleLights() block computes
	// degenerates to exactly 1.0 for every sample from this function. Callers
	// must NOT apply a lightPdf division or an MIS blend against bsdfPdf here
	// - see SampleLights' punctual-light block in Renderer.cpp/
	// SampleLights.slang for the one place in this codebase that
	// deliberately skips the usual MIS math.
	CUDA_CALLABLE inline void PunctualLightSample(const PunctualLight &light, const Vec3 &surfacePos, Random &rand,
												   Vec3 &wi, Vec3 &radiance, float &dist)
	{
		if (light.type == PunctualLightType::ePoint)
		{
			Vec3 samplePos = light.position;
			if (light.radius > 0.0f)
			{
				float u1, u2;
				Sample2D(rand, u1, u2);
				samplePos += UniformSampleSphere(u1, u2) * light.radius;
			}

			Vec3 toLight = samplePos - surfacePos;
			float dSq = LengthSq(toLight);
			dist = sqrtf(dSq);
			wi = dist > 0.0f ? toLight / dist : Vec3(0.0f, 1.0f, 0.0f);

			// Inverse-square falloff - the defining behavior of a point
			// light, unlike the no-falloff directional case below.
			radiance = light.color * light.intensity / Max(dSq, 1.e-6f);
		}
		else // eDirectional
		{
			Vec3 dir = Normalize(light.direction);
			wi = -dir;

			if (light.angle > 0.0f)
			{
				// Jitter within a small cone around -dir for a soft-sun
				// approximation - same "basis + disc offset, renormalize"
				// approach used elsewhere for small-angle perturbation
				// (see NormalMapping.h's tangent re-orthonormalization).
				Vec3 u, v;
				BasisFromVector(wi, &u, &v);

				float u1, u2;
				Sample2D(rand, u1, u2);
				Vec2 disc = UniformSampleDisc(u1, u2) * tanf(light.angle * 0.5f);
				wi = Normalize(wi + u * disc.x + v * disc.y);
			}

			// No distance falloff - a directional light is infinitely far
			// away by definition. A large sentinel distance both places the
			// shadow ray's endpoint effectively at infinity and lets the
			// existing TraceWithCutout occlusion-distance check
			// (Renderer.cpp's SampleLights, `fabsf(t - dist) <= kTolerance`)
			// work unmodified: any real occluder hit will report t << dist.
			dist = 1.e8f;
			radiance = light.color * light.intensity;
		}
	}
}
