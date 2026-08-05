#pragma once

// NOTE: this header was dead code (never #included anywhere) until the
// Preetham sky bake feature (BakePreethamSky, below) started calling
// SkyLight() - its original includes (`mirage/Vec4.h`, a path that doesn't
// exist) had never actually been compiled. Fixed to the real paths for
// Vec4/Mat44/Color, all used by SkyLight() below.
#include "mirage/math/Vec4.h"
#include "mirage/math/Mat44.h"
#include "mirage/math/Color.h"
#include "mirage/utils/MathUtils.h"
#include "mirage/core/Probe.h"

namespace Mirage
{
    // implements Perez's model for sky luminance
    inline float SkyDistribution(float theta, float gamma, float a, float b, float c, float d, float e)
    {
        float cosGamma = cosf(gamma);
        float cosTheta = cosf(theta);

        return (1.0f + a * expf(b / cosTheta)) * (1.0f + c * expf(d * gamma) + e * cosGamma * cosGamma);
    }

    inline float SkyLuminance(float theta, float gamma, float zenith, float sunTheta, float a, float b, float c, float d, float e)
    {
        float l = zenith * (SkyDistribution(theta, gamma, a, b, c, d, e) / SkyDistribution(0.0f, sunTheta, a, b, c, d, e));
        return l;
    }

    inline float Lerp2(const float ab[2], float t)
    {
        return ab[0] * t + ab[1];
    }

    inline Color SkyLight(float theta, float phi, float sunTheta, float sunPhi, float t)
    {
        // need cosTheta > 0.0
        theta = Clamp(theta, 0.0f, kPi * 0.5f - 1.0e-6f);

        // calculate arc-length between sun and patch being calculated
        const float gamma = acosf(cosf(sunTheta) * cosf(theta) + sinf(sunTheta) * sinf(theta) * cosf(abs(phi - sunPhi)));

        const float xcoeff[5][2] = {{-0.0193f, -0.2592f},
                                    {-0.0665f, 0.0008f},
                                    {-0.0004f, 0.2125f},
                                    {-0.0641f, -0.8989f},
                                    {-0.0033f, 0.0452f}};

        const float ycoeff[5][2] = {{-0.0167f, -0.2608f},
                                    {-0.0950f, 0.0092f},
                                    {-0.0079f, 0.2102f},
                                    {-0.0441f, -1.6537f},
                                    {-0.0109f, 0.0529f}};

        const float Ycoeff[5][2] = {{0.1787f, -1.4630f},
                                    {-0.3554f, 0.4275f},
                                    {-0.0227f, 5.3251f},
                                    {0.1206f, -2.5771f},
                                    {-0.0670f, 0.3703f}};

        const Mat44 zxcoeff(0.00166f, -0.02903f, 0.11693f, 0.0f,
                            -0.00375f, 0.06377f, -0.21196f, 0.0f,
                            0.00209f, -0.03202f, 0.06052f, 0.0f,
                            0.0f, 0.00394f, 0.25886f, 0.0f);

        const Mat44 zycoeff(0.00275f, -0.04214f, 0.15346f, 0.0f,
                            -0.00610f, 0.08970f, -0.26756f, 0.0f,
                            0.00317f, -0.04153f, 0.06670f, 0.0f,
                            0.0f, 0.00516f, 0.26688f, 0.0f);

        const Vec4 b(sunTheta * sunTheta * sunTheta, sunTheta * sunTheta, sunTheta, 1.0f);
        const Vec4 a(t * t, t, 1.0f, 0.0f);

        // calculate the zenith values for current turbidity and sun position
        const float zx = Dot(a, zxcoeff * b);
        const float zy = Dot(a, zycoeff * b);
        const float zY = (4.0453f * t - 4.9710f) * tanf((4.0f / 9.0f - t / 120.0f) * (kPi - 2.0f * sunTheta)) - 0.2155f * t + 2.4192f;

        float x = SkyLuminance(theta, gamma, zx, sunTheta, Lerp2(xcoeff[0], t),
                               Lerp2(xcoeff[1], t),
                               Lerp2(xcoeff[2], t),
                               Lerp2(xcoeff[3], t),
                               Lerp2(xcoeff[4], t));

        float y = SkyLuminance(theta, gamma, zy, sunTheta, Lerp2(ycoeff[0], t),
                               Lerp2(ycoeff[1], t),
                               Lerp2(ycoeff[2], t),
                               Lerp2(ycoeff[3], t),
                               Lerp2(ycoeff[4], t));

        float Y = SkyLuminance(theta, gamma, zY, sunTheta, Lerp2(Ycoeff[0], t),
                               Lerp2(Ycoeff[1], t),
                               Lerp2(Ycoeff[2], t),
                               Lerp2(Ycoeff[3], t),
                               Lerp2(Ycoeff[4], t));

        // convert Yxy to XYZ and then to RGB
        Color XYZ = YxyToXYZ(Y, x, y);
        Color RGB = XYZToLinear(XYZ.x, XYZ.y, XYZ.z);

        return RGB;
    }

    // Bakes the Preetham analytic sky model (SkyLight() above) into a
    // lat-long HDR Probe, once, CPU-side, rather than porting the model to
    // Slang - this lets both the CPU and Vulkan/GPU backends reuse the
    // already-working, already-CPU/GPU-parity-tested Probe/ProbeSample/
    // ProbeEval/ProbePdf importance-sampling machinery completely unchanged
    // (see Scene::Sky, VulkanRenderer::UploadProbe, Sky.slang), instead of
    // duplicating this 15-coefficient analytic model in a second language.
    //
    // sunDirection need not be normalized. turbidity is the standard
    // Preetham atmospheric turbidity parameter (~2 = very clear sky,
    // ~6-10 = hazy/overcast-tending). Texels below the horizon (dir.y < 0)
    // are left black - this is a sky-only model, not a full ground renderer.
    inline Probe BakePreethamSky(const Vec3 &sunDirection, float turbidity, int width = 256, int height = 128)
    {
        Vec3 sunDir = SafeNormalize(sunDirection, Vec3(0.0f, 1.0f, 0.0f));
        float sunTheta = acosf(::Mirage::Clamp(sunDir.y, -1.0f, 1.0f));
        float sunPhi = (sunDir.x == 0.0f && sunDir.z == 0.0f) ? 0.0f : atan2f(sunDir.z, sunDir.x);
        // SkyLight()'s analytic model divides by cosTheta-derived terms that
        // are singular at/below the horizon - clamp just above it, the same
        // convention SkyLight() itself uses for the view-direction theta
        // just below.
        sunTheta = ::Mirage::Min(sunTheta, kPi * 0.5f - 1.0e-3f);

        Probe probe;
        probe.width = width;
        probe.height = height;
        probe.data = new Color[width * height];
        probe.offset = Vec3(0.0f);

        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < width; ++i)
            {
                Vec2 uv((i + 0.5f) / float(width), (j + 0.5f) / float(height));
                Vec3 dir = ProbeUVToDir(uv);

                Color c(0.0f, 0.0f, 0.0f, 1.0f);
                if (dir.y > 0.0f)
                {
                    float theta = acosf(::Mirage::Clamp(dir.y, -1.0f, 1.0f));
                    float phi = (dir.x == 0.0f && dir.z == 0.0f) ? 0.0f : atan2f(dir.z, dir.x);
                    Color rgb = SkyLight(theta, phi, sunTheta, sunPhi, turbidity);
                    // The Preetham fit can dip slightly negative near the
                    // horizon at some turbidity/sun-angle combinations -
                    // clamp rather than let a negative radiance propagate
                    // into the path tracer.
                    c = Color(::Mirage::Max(rgb.x, 0.0f), ::Mirage::Max(rgb.y, 0.0f), ::Mirage::Max(rgb.z, 0.0f), 1.0f);
                }

                probe.data[j * width + i] = c;
            }
        }

        probe.BuildCDF();
        return probe;
    }
}
