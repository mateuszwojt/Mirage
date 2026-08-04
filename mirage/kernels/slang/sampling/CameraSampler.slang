// Port of utils/Util.h's CameraSampler::GenerateRay. rasterToWorld/origin are
// precomputed on the host (mirrors the CPU CameraSampler's constructor output)
// to avoid porting quaternion camera-transform math to the GPU - see
// GpuLayout.h's GpuCamera.

import Random;
import Constants;

struct GpuCamera
{
    float4x4 rasterToWorld; // column-major, matches math/Mat44.h's cols[4][4]
    float3 origin;
    float aperture;
    float focalPoint;
    uint enableDOF;
    float shutterStart, shutterEnd;
};

void CameraSampler_GenerateRay(GpuCamera cam, float rasterX, float rasterY, inout Random rand,
                                out float3 origin, out float3 dir)
{
    float3 p = mul(float4(rasterX, rasterY, 0.0f, 1.0f), cam.rasterToWorld).xyz;

    origin = cam.origin;
    dir = normalize(p - origin);

    if (cam.enableDOF != 0u)
    {
        float ox = rand.Randf();
        float oy = rand.Randf();

        float a = 2.0f * ox - 1.0f;
        float b = 2.0f * oy - 1.0f;

        float r, phi;
        if ((a * a) > (b * b))
        {
            r = a;
            phi = 0.78539816339f * (b / a);
        }
        else
        {
            r = b;
            phi = (kPi * 0.5f) - 0.78539816339f * (a / b);
        }

        float2 lens = float2(r * cos(phi), r * sin(phi)) * (cam.aperture * 0.5f);

        float3 w = origin - dir;
        float3 u = cross(float3(0.0f, 1.0f, 0.0f), w);
        float3 v = cross(w, u);
        float3 offset = u * lens.x + v * lens.y;

        origin += offset;

        float3 focusPoint = cam.focalPoint * dir;
        dir = normalize(focusPoint - offset);
    }
}
