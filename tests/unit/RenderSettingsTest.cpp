#include <doctest/doctest.h>

#include "mirage/core/RenderSettings.h"

using namespace Mirage;

TEST_SUITE("RenderSettings")
{
    // Pins the "no render_settings block in the scene file" legacy path
    // (docs/PRODUCTION_READINESS.md's Tier 3 render-settings-map item) -
    // MakeLegacyRenderSettings is what tools/scene_renderer/SceneRenderer.cpp
    // falls back to when a scene file has no render_settings { } blocks, and
    // must keep producing exactly one pass/one product from the
    // positional-args-only call pattern every pre-existing invocation uses.
    TEST_CASE("MakeLegacyRenderSettings produces exactly one pass with one product")
    {
        Options options{};
        options.width = 640;
        options.height = 480;
        options.maxSamples = 16;
        options.exposure = 2.0f;

        RenderSettings settings = MakeLegacyRenderSettings(options, "out.png", kAovDepth, ViewTransform::eAcesLike);

        REQUIRE(settings.size() == 1);
        REQUIRE(settings[0].products.size() == 1);
    }

    TEST_CASE("MakeLegacyRenderSettings's pass carries through every shared Options field unmodified")
    {
        Options options{};
        options.width = 640;
        options.height = 480;
        options.maxSamples = 16;
        options.maxDepth = 5;
        options.exposure = 2.0f;
        options.clamp = 10.0f;
        options.enableDOF = true;
        options.accumulateAovs = true;
        options.enableDenoise = true;
        options.nlmWidth = 3.0f;
        options.nlmFalloff = 0.5f;

        RenderSettings settings = MakeLegacyRenderSettings(options, "out.exr", 0, ViewTransform::eFilmic);

        const Options &passOptions = settings[0].options;
        CHECK(passOptions.width == 640);
        CHECK(passOptions.height == 480);
        CHECK(passOptions.maxSamples == 16);
        CHECK(passOptions.maxDepth == 5);
        CHECK(passOptions.exposure == doctest::Approx(2.0f));
        CHECK(passOptions.clamp == doctest::Approx(10.0f));
        CHECK(passOptions.enableDOF == true);
        CHECK(passOptions.accumulateAovs == true);
        CHECK(passOptions.enableDenoise == true);
        CHECK(passOptions.nlmWidth == doctest::Approx(3.0f));
        CHECK(passOptions.nlmFalloff == doctest::Approx(0.5f));
    }

    TEST_CASE("MakeLegacyRenderSettings's aovMask is applied to both the pass's Options and the product")
    {
        Options options{};
        uint32_t aovMask = kAovDepth | kAovAlbedo;

        RenderSettings settings = MakeLegacyRenderSettings(options, "out.exr", aovMask, ViewTransform::eNone);

        // The pass-level Options::aovMask is what actually drives Render()'s
        // AOV capture - it must match the one product's request, or the
        // product's aovMask bits would silently go unpopulated.
        CHECK(settings[0].options.aovMask == aovMask);
        CHECK(settings[0].products[0].aovMask == aovMask);
    }

    TEST_CASE("MakeLegacyRenderSettings's product carries the requested name, path, and view transform")
    {
        Options options{};
        RenderSettings settings = MakeLegacyRenderSettings(options, "render/out.exr", kAovNormal, ViewTransform::eSrgbDisplay);

        const RenderProduct &product = settings[0].products[0];
        CHECK(product.name == "beauty");
        CHECK(product.outputPath == "render/out.exr");
        CHECK(product.viewTransform == ViewTransform::eSrgbDisplay);
    }

    TEST_CASE("MakeLegacyRenderSettings defaults aovMask to 0 (color-only) when not requested")
    {
        Options options{};
        RenderSettings settings = MakeLegacyRenderSettings(options, "out.png", 0, ViewTransform::eFilmic);

        CHECK(settings[0].options.aovMask == 0u);
        CHECK(settings[0].products[0].aovMask == 0u);
    }
}
