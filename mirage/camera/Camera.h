#pragma once

#include "mirage/math/Vec3.h"
#include "mirage/math/Mat33.h"

namespace Mirage
{
	class Camera
	{
	public:
		Camera() : fov(::Mirage::DegToRad(35.0f)), shutterStart(0.0f), shutterEnd(1.0f) {};
		Camera(float fov, float shutterStart, float shutterEnd);

		Vec3 position;
		Quat rotation;

		// Legacy/non-physical field of view (radians). Used verbatim by
		// EffectiveFov() whenever focalLength <= 0 - the physical fields below
		// are opt-in.
		float fov;

		// Normalized [0,1] motion-blur shutter window fraction - NOT a physical
		// exposure time. See shutterSpeed below for that.
		float shutterStart;
		float shutterEnd;

		// Legacy/non-physical lens aperture diameter and focal (focus) distance,
		// both in scene-world units. Used verbatim by EffectiveApertureDiameter()
		// whenever fStop/focalLength are unset. 0 = "unset" is not a meaningful
		// aperture, so these default to 0 (no DOF) rather than being left
		// uninitialized.
		float aperture = 0.0f;
		float focalPoint = 1.0f;

		// --- Physical camera parameters (all optional/opt-in; 0 = unset) ---

		// Lens focal length in millimeters. When > 0, EffectiveFov() derives the
		// vertical field of view from this + sensorHeight instead of using the
		// legacy `fov` field.
		float focalLength = 0.0f;

		// Sensor dimensions in millimeters (full-frame 35mm default).
		float sensorWidth = 36.0f;
		float sensorHeight = 24.0f;

		// Relative aperture (f-number). When > 0 (and focalLength > 0),
		// EffectiveApertureDiameter() derives a physical aperture diameter from
		// focalLength/fStop instead of using the legacy `aperture` field.
		float fStop = 0.0f;

		// Physical exposure time in SECONDS - distinct from shutterStart/
		// shutterEnd above, which are a normalized motion-blur sampling window,
		// not a real-world duration. When this, iso, and fStop are all set,
		// ComputeExposureMultiplier() derives an EV100-based exposure multiplier.
		float shutterSpeed = 0.0f;

		// Sensor sensitivity (ISO 100 default).
		float iso = 100.0f;

		// Resolves the vertical FOV (radians) to use for ray generation: derived
		// from focalLength/sensorHeight if focalLength is set, otherwise the
		// legacy `fov` field unchanged.
		float EffectiveFov() const
		{
			if (focalLength > 0.0f)
			{
				return 2.0f * atanf(sensorHeight / (2.0f * focalLength));
			}
			return fov;
		}

		// Resolves the lens aperture diameter (scene-world units) to use for
		// depth-of-field sampling: derived from focalLength/fStop (converted
		// mm -> scene-world meters) if both are set, otherwise the legacy
		// `aperture` field unchanged.
		float EffectiveApertureDiameter() const
		{
			if (fStop > 0.0f && focalLength > 0.0f)
			{
				return (focalLength / fStop) / 1000.0f;
			}
			return aperture;
		}

		// Resolves a physically-derived exposure multiplier from shutterSpeed/
		// iso/fStop (standard EV100 formula), or 1.0 (no-op) if any of the three
		// are unset - i.e. a scene that never sets physical exposure parameters
		// renders with zero behavior change.
		float ComputeExposureMultiplier() const
		{
			if (shutterSpeed > 0.0f && iso > 0.0f && fStop > 0.0f)
			{
				float ev = log2f((fStop * fStop) / shutterSpeed) - log2f(iso / 100.0f);
				return 1.0f / (powf(2.0f, ev) * 1.2f);
			}
			return 1.0f;
		}
	};
}