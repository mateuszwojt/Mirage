#include <doctest/doctest.h>

#include "mirage/math/Color.h"

using namespace Mirage;

TEST_SUITE("Color")
{
    // LinearToSrgb/SrgbToLinear currently use a naive pow(c, 1/2.2)/pow(c, 2.2)
    // gamma approximation (mirage/math/Color.h:102-112), not a proper piecewise
    // sRGB EOTF/OETF (see the correct implementation in
    // mirage/shaders/TextureLoader.cpp:13-18 for comparison, and
    // docs/PRODUCTION_READINESS.md's Tier 3 "no color management" item for the
    // plan to unify them). This test pins today's actual round-trip behavior
    // so that unification is a deliberate, visible change, not a silent drift.
    TEST_CASE("LinearToSrgb and SrgbToLinear round-trip (current pow-2.2 approximation)")
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

    TEST_CASE("LinearToSrgb brightens mid-tones (gamma < 1 exponent)")
    {
        const Color linear(0.18f, 0.18f, 0.18f, 1.0f);
        const Color srgb = LinearToSrgb(linear);

        CHECK(srgb.x > linear.x);
        CHECK(srgb.x == doctest::Approx(powf(0.18f, 1.0f / 2.2f)));
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
