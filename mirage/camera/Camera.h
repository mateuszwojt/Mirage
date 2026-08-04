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

		float fov;

		float shutterStart;
		float shutterEnd;

		float aperture;
		float focalPoint;
	};
}