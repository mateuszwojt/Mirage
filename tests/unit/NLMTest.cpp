#include <doctest/doctest.h>

#include "mirage/filter/NLM.h"

using namespace Mirage;

namespace
{
    // Deterministic, dependency-free pseudo-noise (no <random> seeding
    // dance needed for a simple, reproducible test fixture) - not
    // statistically rigorous, just "not all the same value".
    float PseudoNoise(int i)
    {
        int h = i * 2654435761u;
        return static_cast<float>(h & 0xFFFF) / 65535.0f; // [0, 1)
    }
} // namespace

TEST_SUITE("NonLocalMeansFilter")
{
    TEST_CASE("denoising a flat color with additive noise reduces variance")
    {
        const int width = 32, height = 32;
        const int n = width * height;

        std::vector<Color> noisy(n);
        for (int i = 0; i < n; ++i)
        {
            float noise = (PseudoNoise(i) - 0.5f) * 0.4f; // +-0.2 around 0.5
            noisy[i] = Color(0.5f + noise, 0.5f + noise, 0.5f + noise, 1.0f);
        }

        std::vector<Color> denoised(n);
        NonLocalMeansFilter(noisy.data(), denoised.data(), width, height, /*falloff=*/50.0f, /*radius=*/3);

        auto varianceOf = [&](const std::vector<Color> &buf) {
            double mean = 0.0;
            for (const Color &c : buf)
                mean += c.x;
            mean /= n;

            double variance = 0.0;
            for (const Color &c : buf)
                variance += (c.x - mean) * (c.x - mean);
            return variance / n;
        };

        double noisyVariance = varianceOf(noisy);
        double denoisedVariance = varianceOf(denoised);

        CHECK(denoisedVariance < noisyVariance);
    }

    TEST_CASE("denoising a perfectly flat image is a no-op (up to floating point)")
    {
        const int width = 8, height = 8;
        const int n = width * height;

        std::vector<Color> flat(n, Color(0.3f, 0.4f, 0.5f, 1.0f));
        std::vector<Color> out(n);

        NonLocalMeansFilter(flat.data(), out.data(), width, height, 50.0f, 2);

        for (const Color &c : out)
        {
            CHECK(c.x == doctest::Approx(0.3f));
            CHECK(c.y == doctest::Approx(0.4f));
            CHECK(c.z == doctest::Approx(0.5f));
        }
    }

    // Guide buffers (albedo/normal) default to nullptr, reducing to plain
    // radiance-only NLM - this pins that "no guide buffers passed" produces
    // identical output to the explicit two-arg call, since a caller that
    // hasn't been updated for the new optional params must see unchanged
    // behavior.
    TEST_CASE("omitting guide buffers matches passing explicit nullptrs")
    {
        const int width = 16, height = 16;
        const int n = width * height;

        std::vector<Color> noisy(n);
        for (int i = 0; i < n; ++i)
        {
            float noise = (PseudoNoise(i) - 0.5f) * 0.4f;
            noisy[i] = Color(0.5f + noise, 0.5f + noise, 0.5f + noise, 1.0f);
        }

        std::vector<Color> outDefault(n), outExplicitNull(n);
        NonLocalMeansFilter(noisy.data(), outDefault.data(), width, height, 50.0f, 3);
        NonLocalMeansFilter(noisy.data(), outExplicitNull.data(), width, height, 50.0f, 3, nullptr, nullptr);

        for (int i = 0; i < n; ++i)
        {
            CHECK(outDefault[i].x == doctest::Approx(outExplicitNull[i].x));
            CHECK(outDefault[i].y == doctest::Approx(outExplicitNull[i].y));
            CHECK(outDefault[i].z == doctest::Approx(outExplicitNull[i].z));
        }
    }

    // A guide buffer with a hard edge (two flat halves of very different
    // albedo) should suppress cross-edge blending relative to plain
    // radiance-only NLM - the whole point of adding it. Uses noisy radiance
    // that's otherwise IDENTICAL on both sides of the edge, so any
    // difference in the denoised result at the edge is attributable to the
    // guide buffer, not a radiance difference the un-guided filter would
    // have picked up on anyway.
    TEST_CASE("an albedo guide buffer preserves a radiance-invisible edge that plain NLM would blur")
    {
        const int width = 16, height = 16;
        const int n = width * height;

        std::vector<Color> noisy(n);
        std::vector<Color> albedo(n);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int i = y * width + x;
                float noise = (PseudoNoise(i) - 0.5f) * 0.1f;
                // Same noisy radiance everywhere - no edge visible in `noisy`
                // itself.
                noisy[i] = Color(0.5f + noise, 0.5f + noise, 0.5f + noise, 1.0f);
                // Hard vertical albedo edge at the midline.
                float a = (x < width / 2) ? 0.1f : 0.9f;
                albedo[i] = Color(a, a, a, 1.0f);
            }
        }

        std::vector<Color> plain(n), guided(n);
        NonLocalMeansFilter(noisy.data(), plain.data(), width, height, 50.0f, 4);
        NonLocalMeansFilter(noisy.data(), guided.data(), width, height, 50.0f, 4, albedo.data(), nullptr);

        // Pixel just left of the midline, comparing against a same-side
        // pixel far from the edge: with a strong albedo guide, this pixel's
        // denoised value should stay close to its same-side neighborhood
        // (guide-suppressed cross-edge contribution), whereas plain NLM
        // (no notion of the edge at all) treats every neighbor in `radius`
        // identically. This checks the guided filter's edge-adjacent pixel
        // differs from the plain filter's - i.e. the guide buffer actually
        // changed the outcome - not an exact target value (the precise
        // number depends on the neighborhood weighting math).
        int edgeAdjacentIdx = 8 * width + (width / 2 - 1);
        CHECK(guided[edgeAdjacentIdx].x != doctest::Approx(plain[edgeAdjacentIdx].x));
    }
}
