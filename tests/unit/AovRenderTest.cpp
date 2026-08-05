#include <doctest/doctest.h>

#include "mirage/core/Renderer.h"
#include "mirage/camera/Camera.h"

using namespace Mirage;

namespace
{
    // One sphere, centered on-axis, filling most of a small frame - deliberately
    // minimal, just enough to exercise real primary-ray hit/miss AOV capture
    // end to end (mirage/core/Renderer.cpp's CpuRenderer::Render()/PathTrace()),
    // not just the AovBuffers/NamedAov data-holder contract AovTest.cpp checks.
    void BuildOneSphereScene(Scene &scene)
    {
        Primitive sphere;
        sphere.type = eSphere;
        sphere.sphere.radius = 1.0f;
        sphere.startTransform = Transform(Vec3(0.0f, 0.0f, 0.0f));
        sphere.endTransform = sphere.startTransform;

        auto mat = std::make_unique<Material>();
        mat->color = Vec3(0.8f, 0.2f, 0.1f);
        sphere.materialIndex = scene.AddMaterial(std::move(mat));

        scene.AddPrimitive(sphere);
        scene.Build();
    }
} // namespace

TEST_SUITE("AOV capture (real CPU render)")
{
    TEST_CASE("kAovAlbedo reports the hit material's color on a hit, zero on a miss")
    {
        const int width = 16, height = 16;

        Scene scene;
        BuildOneSphereScene(scene);

        Camera camera;
        camera.position = Vec3(0.0f, 0.0f, 4.0f);
        camera.fov = DegToRad(45.0f);

        Options options{};
        options.mode = ePathTrace;
        options.type = eCpu;
        options.width = width;
        options.height = height;
        options.enableDOF = false;
        options.maxDepth = 1;
        options.maxSamples = 1;
        options.exposure = 1.0f;
        options.limit = 1000.0f;
        options.clamp = 20.0f;
        options.aovMask = kAovAlbedo;

        std::vector<Color> beauty(width * height, Color(0.0f));
        std::vector<Color> albedo(width * height, Color(0.0f));
        AovBuffers aovs;
        aovs.albedo = albedo.data();

        std::unique_ptr<Renderer> renderer(CreateCpuRenderer(&scene));
        renderer->Render(camera, options, beauty.data(), &aovs);

        // Center pixel: the sphere fills most of a 45deg-FOV frame at this
        // distance, so the center ray should hit it.
        const Color &center = albedo[(height / 2) * width + (width / 2)];
        CHECK(center.x == doctest::Approx(0.8f).epsilon(0.05));
        CHECK(center.y == doctest::Approx(0.2f).epsilon(0.05));
        CHECK(center.z == doctest::Approx(0.1f).epsilon(0.05));

        // Corner pixel: outside the sphere's silhouette at this FOV/distance,
        // so the primary ray misses everything - default-constructed Scene
        // has no sky, so a miss should read back as the outAlbedo sentinel
        // (Vec3(0.0f), see PathTrace()'s capture defaults), not garbage.
        const Color &corner = albedo[0];
        CHECK(corner.x == doctest::Approx(0.0f));
        CHECK(corner.y == doctest::Approx(0.0f));
        CHECK(corner.z == doctest::Approx(0.0f));
    }

    TEST_CASE("Named AOV \"P\" reports a world-space hit position on a hit, zero on a miss")
    {
        const int width = 16, height = 16;

        Scene scene;
        BuildOneSphereScene(scene);

        Camera camera;
        camera.position = Vec3(0.0f, 0.0f, 4.0f);
        camera.fov = DegToRad(45.0f);

        Options options{};
        options.mode = ePathTrace;
        options.type = eCpu;
        options.width = width;
        options.height = height;
        options.enableDOF = false;
        options.maxDepth = 1;
        options.maxSamples = 1;
        options.exposure = 1.0f;
        options.limit = 1000.0f;
        options.clamp = 20.0f;
        // aovMask deliberately left 0 - a NamedAov-only request (no AovType
        // bits set) must still trigger capture, exercising Render()'s
        // haveNamedAovs half of the wantAovs gate.
        options.aovMask = 0;

        std::vector<Color> beauty(width * height, Color(0.0f));
        std::vector<Color> positionBuf(width * height, Color(0.0f));
        AovBuffers aovs;
        aovs.named.push_back(NamedAov{"P", positionBuf.data()});

        std::unique_ptr<Renderer> renderer(CreateCpuRenderer(&scene));
        renderer->Render(camera, options, beauty.data(), &aovs);

        // Center hit: world-space position on a unit sphere centered at the
        // origin, camera looking down -Z from (0,0,4) - the hit point should
        // be close to (0, 0, 1), the near pole of the sphere.
        const Color &center = positionBuf[(height / 2) * width + (width / 2)];
        CHECK(center.x == doctest::Approx(0.0f).epsilon(0.2));
        CHECK(center.y == doctest::Approx(0.0f).epsilon(0.2));
        CHECK(center.z == doctest::Approx(1.0f).epsilon(0.2));

        // Miss: sentinel zero, same reasoning as the albedo test above.
        const Color &corner = positionBuf[0];
        CHECK(corner.x == doctest::Approx(0.0f));
        CHECK(corner.y == doctest::Approx(0.0f));
        CHECK(corner.z == doctest::Approx(0.0f));
    }
}
