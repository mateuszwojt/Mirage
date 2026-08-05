#pragma once

// Render-settings-map / multi-product batch pipeline (docs/PRODUCTION_READINESS.md's
// Tier 3 "no render-settings-map / multi-product batch pipeline" finding):
// Options is a single fixed render configuration, with no concept of
// multiple named output products (beauty + individual AOV files, different
// view transforms of the same render, etc.) coming out of one invocation.
// RenderProduct/RenderPass/RenderSettings below are the data model for that,
// used by tools/scene_renderer/SceneRenderer.cpp to drive both a
// backward-compatible single-file legacy invocation (MakeLegacyRenderSettings)
// and a scene file's repeatable `render_settings { }` blocks.

#include "mirage/core/Renderer.h"
#include "mirage/utils/Util.h" // ViewTransform

#include <string>
#include <utility>
#include <vector>

namespace Mirage
{
    // One named output file, sharing its RenderPass's accumulated buffers -
    // producing N products from a single RenderPass costs one Render() call,
    // not N.
    struct RenderProduct
    {
        std::string name;       // e.g. "beauty", "diffuse_aov" - informational, not used to derive outputPath
        std::string outputPath;
        uint32_t aovMask = 0;   // 0 = beauty only, bitwise-or of AovType (see Renderer.h)
        ViewTransform viewTransform = ViewTransform::eFilmic;
    };

    // One Render() call's worth of shared configuration (resolution, sample
    // count, depth, filter, ...), plus every RenderProduct it feeds. Products
    // needing an independent resolution or sample count belong in their own
    // RenderPass, not this one - see BuildRenderSettingsFromDefs in
    // SceneRenderer.cpp for the grouping logic that decides that.
    struct RenderPass
    {
        Options options;
        std::vector<RenderProduct> products;
    };

    // A full render-settings-map: every pass a `scene_renderer` invocation
    // will execute, in order.
    using RenderSettings = std::vector<RenderPass>;

    // Synthesizes the single-pass/single-product RenderSettings equivalent to
    // every pre-existing scene_renderer invocation (a scene file with no
    // `render_settings { }` blocks) - this is what keeps that positional-
    // args-only call pattern working unchanged after this feature landed.
    // `options` should already have every shared field set (width/height/
    // maxSamples/maxDepth/filter/exposure/clamp/enableDOF/accumulateAovs/
    // enableDenoise/nlmWidth/nlmFalloff/type/mode) - this only overwrites
    // options.aovMask (from `aovMask`, so the one RenderPass's Render() call
    // populates exactly what the one RenderProduct needs) and wraps the
    // single RenderProduct/RenderPass/RenderSettings layers around it.
    inline RenderSettings MakeLegacyRenderSettings(Options options, const std::string &outputPath, uint32_t aovMask,
                                                     ViewTransform viewTransform)
    {
        options.aovMask = aovMask;

        RenderProduct product;
        product.name = "beauty";
        product.outputPath = outputPath;
        product.aovMask = aovMask;
        product.viewTransform = viewTransform;

        RenderPass pass;
        pass.options = std::move(options);
        pass.products.push_back(std::move(product));

        RenderSettings settings;
        settings.push_back(std::move(pass));
        return settings;
    }
}
