#include <doctest/doctest.h>

#include "mirage/core/Renderer.h"

using namespace Mirage;

TEST_SUITE("AOV bitmask")
{
    // Pins the current 3-bit AovType layout (mirage/core/Renderer.h:67-72) so
    // future extensions (see docs/PRODUCTION_READINESS.md's Tier 3 AOV item -
    // albedo, motion vectors, arbitrary/named AOVs) add new bits rather than
    // silently renumbering existing ones, which would be a binary-compatibility
    // break for any external caller (e.g. the Hydra delegate) storing aovMask
    // values.
    TEST_CASE("AovType bits are distinct, non-overlapping powers of two")
    {
        CHECK(kAovDepth == 1u);
        CHECK(kAovNormal == 2u);
        CHECK(kAovPrimId == 4u);

        CHECK((kAovDepth & kAovNormal) == 0u);
        CHECK((kAovDepth & kAovPrimId) == 0u);
        CHECK((kAovNormal & kAovPrimId) == 0u);
    }

    TEST_CASE("Options::aovMask defaults to 0 (color-only, preserving every pre-existing call site)")
    {
        Options options{};
        CHECK(options.aovMask == 0u);
    }

    TEST_CASE("aovMask combines via bitwise-or and each bit is independently testable")
    {
        uint32_t mask = kAovDepth | kAovPrimId;

        CHECK((mask & kAovDepth) != 0u);
        CHECK((mask & kAovNormal) == 0u);
        CHECK((mask & kAovPrimId) != 0u);
    }

    TEST_CASE("AovBuffers defaults every pointer to null (caller must opt in per-buffer)")
    {
        AovBuffers aovs{};
        CHECK(aovs.depth == nullptr);
        CHECK(aovs.normal == nullptr);
        CHECK(aovs.primId == nullptr);
    }
}
