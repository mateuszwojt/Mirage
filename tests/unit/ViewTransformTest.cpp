#include <doctest/doctest.h>

#include "mirage/utils/Util.h"

using namespace Mirage;

TEST_SUITE("ToneMap / ToneMapACES / ApplyViewTransform")
{
    // Confirms the fixed bug directly: ToneMap's curve (Hejl/Burgess-Dawson)
    // is display-encoded on its own - the old code called SrgbToLinear (a
    // *decode*) on that already-encoded output, double-transforming every
    // LDR render. If that regressed, this would start failing (the
    // now-nonexistent decode would darken mid-tones well below what the
    // curve alone produces).
    TEST_CASE("ToneMap of a mid-gray input is not decoded a second time")
    {
        const Color midGray(0.5f, 0.5f, 0.5f, 1.0f);
        const Color mapped = ToneMap(midGray, 1.0f);

        // Ballpark sanity range for this curve at x=0.5, not an exact
        // literal - the point is "not further darkened by an erroneous
        // SrgbToLinear(x^2.2-ish) pass", which would push this well below
        // ~0.3.
        CHECK(mapped.x > 0.4f);
        CHECK(mapped.x < 0.9f);
    }

    TEST_CASE("ToneMap and ToneMapACES preserve alpha (previously hardcoded to 0)")
    {
        const Color c(0.3f, 0.4f, 0.5f, 0.75f);
        CHECK(ToneMap(c, 1.0f).w == doctest::Approx(0.75f));
        CHECK(ToneMapACES(c, 1.0f).w == doctest::Approx(0.75f));
    }

    TEST_CASE("ToneMap maps black to (near) black")
    {
        const Color black(0.0f, 0.0f, 0.0f, 1.0f);
        const Color mapped = ToneMap(black, 1.0f);
        CHECK(mapped.x == doctest::Approx(0.0f).epsilon(0.01));
        CHECK(mapped.y == doctest::Approx(0.0f).epsilon(0.01));
        CHECK(mapped.z == doctest::Approx(0.0f).epsilon(0.01));
    }

    // `limit` was previously a dead parameter (declared, never read). Now
    // wired as a pre-curve highlight-compression control: a larger limit
    // divides the input further down before the curve, pushing the same
    // input further from highlight roll-off (i.e. darker for a bright
    // input).
    TEST_CASE("limit compresses highlights - larger limit darkens a bright input")
    {
        const Color bright(1.0f, 1.0f, 1.0f, 1.0f);
        const Color tight = ToneMap(bright, 1.0f);
        const Color loose = ToneMap(bright, 4.0f);

        CHECK(loose.x < tight.x);
    }

    TEST_CASE("ApplyViewTransform(eNone) applies exposure with no display encode")
    {
        const Color linear(0.25f, 0.25f, 0.25f, 1.0f);
        const Color result = ApplyViewTransform(linear, ViewTransform::eNone, 2.0f);

        CHECK(result.x == doctest::Approx(0.5f));
        CHECK(result.y == doctest::Approx(0.5f));
        CHECK(result.z == doctest::Approx(0.5f));
    }

    TEST_CASE("ApplyViewTransform(eSrgbDisplay) matches exposure then LinearToSrgb directly")
    {
        const Color linear(0.5f, 0.5f, 0.5f, 1.0f);
        const Color result = ApplyViewTransform(linear, ViewTransform::eSrgbDisplay, 1.0f);
        const Color expected = LinearToSrgb(linear);

        CHECK(result.x == doctest::Approx(expected.x));
        CHECK(result.y == doctest::Approx(expected.y));
        CHECK(result.z == doctest::Approx(expected.z));
    }

    TEST_CASE("ApplyViewTransform(eFilmic) and (eAcesLike) respond monotonically to exposure")
    {
        const Color linear(0.2f, 0.2f, 0.2f, 1.0f);

        const Color dim = ApplyViewTransform(linear, ViewTransform::eFilmic, 0.5f);
        const Color bright = ApplyViewTransform(linear, ViewTransform::eFilmic, 4.0f);
        CHECK(bright.x > dim.x);

        const Color dimAces = ApplyViewTransform(linear, ViewTransform::eAcesLike, 0.5f);
        const Color brightAces = ApplyViewTransform(linear, ViewTransform::eAcesLike, 4.0f);
        CHECK(brightAces.x > dimAces.x);
    }
}
