#include "mirage/camera/Camera.h"

namespace Mirage
{
    Camera::Camera(float fov, float shutterStart, float shutterEnd) : fov(fov), shutterStart(shutterStart), shutterEnd(shutterEnd)
    {
        position = Mirage::Vec3(0.0f, 1.0f, 5.0f); // some generic value
        rotation = Mirage::Quat();
        fov = Mirage::DegToRad(fov);
    }
}