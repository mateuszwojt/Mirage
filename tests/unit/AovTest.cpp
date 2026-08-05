#include <doctest/doctest.h>

#include "mirage/core/Renderer.h"

using namespace Mirage;

TEST_SUITE("AOV bitmask")
{
    // Pins the current 4-bit AovType layout (mirage/core/Renderer.h) so
    // future extensions (see docs/PRODUCTION_READINESS.md's Tier 3 AOV item -
    // motion vectors, more arbitrary/named AOVs) add new bits rather than
    // silently renumbering existing ones, which would be a binary-compatibility
    // break for any external caller (e.g. the Hydra delegate) storing aovMask
    // values.
    TEST_CASE("AovType bits are distinct, non-overlapping powers of two")
    {
        CHECK(kAovDepth == 1u);
        CHECK(kAovNormal == 2u);
        CHECK(kAovPrimId == 4u);
        CHECK(kAovAlbedo == 8u);

        CHECK((kAovDepth & kAovNormal) == 0u);
        CHECK((kAovDepth & kAovPrimId) == 0u);
        CHECK((kAovDepth & kAovAlbedo) == 0u);
        CHECK((kAovNormal & kAovPrimId) == 0u);
        CHECK((kAovNormal & kAovAlbedo) == 0u);
        CHECK((kAovPrimId & kAovAlbedo) == 0u);
    }

    TEST_CASE("Options::aovMask defaults to 0 (color-only, preserving every pre-existing call site)")
    {
        Options options{};
        CHECK(options.aovMask == 0u);
    }

    TEST_CASE("Options::accumulateAovs defaults to false (first-hit-only, preserving every pre-existing call site)")
    {
        Options options{};
        CHECK(options.accumulateAovs == false);
    }

    TEST_CASE("aovMask combines via bitwise-or and each bit is independently testable")
    {
        uint32_t mask = kAovDepth | kAovPrimId | kAovAlbedo;

        CHECK((mask & kAovDepth) != 0u);
        CHECK((mask & kAovNormal) == 0u);
        CHECK((mask & kAovPrimId) != 0u);
        CHECK((mask & kAovAlbedo) != 0u);
    }

    TEST_CASE("AovBuffers defaults every pointer to null (caller must opt in per-buffer) and named is empty")
    {
        AovBuffers aovs{};
        CHECK(aovs.depth == nullptr);
        CHECK(aovs.normal == nullptr);
        CHECK(aovs.primId == nullptr);
        CHECK(aovs.albedo == nullptr);
        CHECK(aovs.named.empty());
    }

    // Tier B / arbitrary AOVs (NamedAov) - just the plain-data-holder
    // contract (name + buffer pointer), since the actual population logic
    // (mirage/core/Renderer.cpp's PathTrace()/Render(), recognizing "P" and
    // "uv") needs a real render and is covered by kernel_validate/manual
    // testing instead, not a unit test.
    TEST_CASE("NamedAov defaults buffer to null and can be added to AovBuffers::named")
    {
        NamedAov na;
        CHECK(na.buffer == nullptr);
        CHECK(na.name.empty());

        Color buffer[4] = {};
        AovBuffers aovs;
        aovs.named.push_back(NamedAov{"P", buffer});
        aovs.named.push_back(NamedAov{"uv", buffer});

        REQUIRE(aovs.named.size() == 2);
        CHECK(aovs.named[0].name == "P");
        CHECK(aovs.named[0].buffer == buffer);
        CHECK(aovs.named[1].name == "uv");
    }
}
