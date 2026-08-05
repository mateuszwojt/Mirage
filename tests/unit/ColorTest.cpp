#include <doctest/doctest.h>

#include "mirage/math/Color.h"

using namespace Mirage;

TEST_SUITE("Color")
{
    // LinearToSrgb/SrgbToLinear now use the correct piecewise IEC 61966-2-1
    // sRGB OETF/EOTF (mirage/math/Color.h), unified with what used to be a
    // second, private, disagreeing implementation in
    // mirage/shaders/TextureLoader.cpp - see docs/PRODUCTION_READINESS.md's
    // Tier 3 "no color management" item. This test pins the curve against
    // well-known reference points (not just self-round-trip, which a
    // consistent-but-wrong pair of curves could also pass) so a regression
    // back to the naive pow(c, 1/2.2)/pow(c, 2.2) approximation is caught.
    TEST_CASE("LinearToSrgb/SrgbToLinear match known IEC 61966-2-1 reference points")
    {
        // Widely-cited reference values for this exact curve.
        CHECK(SrgbToLinear(0.5f) == doctest::Approx(0.214041f).epsilon(0.001));
        CHECK(LinearToSrgb(0.5f) == doctest::Approx(0.735357f).epsilon(0.001));

        // Both curves pass through the endpoints exactly.
        CHECK(SrgbToLinear(0.0f) == doctest::Approx(0.0f));
        CHECK(SrgbToLinear(1.0f) == doctest::Approx(1.0f));
        CHECK(LinearToSrgb(0.0f) == doctest::Approx(0.0f));
        CHECK(LinearToSrgb(1.0f) == doctest::Approx(1.0f));

        // The linear-segment toe below the piecewise breakpoints (0.04045
        // sRGB / 0.0031308 linear) - a naive pow() curve has no such segment.
        CHECK(SrgbToLinear(0.04045f) == doctest::Approx(0.04045f / 12.92f).epsilon(0.0001));
        CHECK(LinearToSrgb(0.0031308f) == doctest::Approx(0.0031308f * 12.92f).epsilon(0.0001));
    }

    TEST_CASE("LinearToSrgb and SrgbToLinear round-trip")
    {
        const Color linear(0.0f, 0.18f, 0.5f, 1.0f);
        const Color srgb = LinearToSrgb(linear);
        const Color roundTripped = SrgbToLinear(srgb);

        CHECK(roundTripped.x == doctest::Approx(linear.x).epsilon(0.001));
        CHECK(roundTripped.y == doctest::Approx(linear.y).epsilon(0.001));
        CHECK(roundTripped.z == doctest::Approx(linear.z).epsilon(0.001));
        // Alpha (.w) must pass through untouched - it's not a color channel.
        CHECK(roundTripped.w == doctest::Approx(linear.w));
    }

    TEST_CASE("LinearToSrgb brightens mid-tones")
    {
        const Color linear(0.18f, 0.18f, 0.18f, 1.0f);
        const Color srgb = LinearToSrgb(linear);

        CHECK(srgb.x > linear.x);
    }

    TEST_CASE("Luminance matches the documented weighting")
    {
        // mirage/math/Color.h:114-117 - note these weights (0.3/0.6/0.1) are
        // NOT the standard Rec.709 luma weights (0.2126/0.7152/0.0722); this
        // test pins the codebase's actual current behavior, not the "correct"
        // photometric answer.
        const Color c(1.0f, 0.0f, 0.0f, 1.0f);
        CHECK(Luminance(c) == doctest::Approx(0.3f));

        const Color white(1.0f, 1.0f, 1.0f, 1.0f);
        CHECK(Luminance(white) == doctest::Approx(1.0f));
    }

    TEST_CASE("ColorToRGBA8 clamps and quantizes to 8-bit")
    {
        const Color overbright(2.0f, -1.0f, 0.5f, 1.0f);
        const unsigned int packed = ColorToRGBA8(overbright);

        const uint8_t r = packed & 0xFF;
        const uint8_t g = (packed >> 8) & 0xFF;
        const uint8_t b = (packed >> 16) & 0xFF;

        CHECK(r == 255); // clamped from 2.0
        CHECK(g == 0);   // clamped from -1.0
        CHECK(b == doctest::Approx(127).epsilon(1));
    }
}
