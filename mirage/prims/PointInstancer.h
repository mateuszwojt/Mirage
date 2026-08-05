#pragma once

#include "mirage/prims/Primitive.h"

#include <vector>

namespace Mirage
{
    // Procedural point-instancer (Tier-1 "instancing/particles", v1 scope):
    // stamps out one Primitive per entry in instanceStart/instanceEnd, all
    // sharing one template shape (sphere/plane/mesh) and one material -
    // expanded by Scene::Build() (see Scene::ExpandInstancers()) before the
    // scene BVH is built. No new traversal/intersection code on either
    // backend; GPU only ever sees the already-flat, expanded `primitives`
    // list, same as today.
    //
    // Tradeoff, stated explicitly: this buys back the *authoring* gap (no
    // way today to place N copies of one mesh without N handwritten
    // Primitive structs) and the mesh-reference data-model wart, not the
    // BVH memory/traversal scaling gap - 10,000 instances still means
    // 10,000 scene-BVH leaves, identical to authoring them by hand. A real
    // TLAS/BLAS-backed instancing path (shared BVH, transform-array
    // traversal) is a larger, separate follow-up.
    //
    // Per-instance material overrides are explicitly out of v1 scope - every
    // instance shares `materialIndex`/`lightSamples`/`hydraId` below.
    struct PointInstancer
    {
        Type type = eSphere;

        // Only the field matching `type` is read by ExpandInstancers().
        float sphereRadius = 1.0f;
        float plane[4] = {0.0f, 1.0f, 0.0f, 0.0f};
        int32_t meshIndex = -1; // index into Scene::meshes - type == eMesh only
        float rectWidth = 1.0f;
        float rectHeight = 1.0f;
        float diskRadius = 1.0f;

        int32_t materialIndex = -1;
        int lightSamples = 0;
        int32_t hydraId = -1;

        // One entry per instance. instanceEnd, if non-empty, must be the
        // same size as instanceStart (one end-of-shutter transform per
        // start-of-shutter one, enabling rigid per-instance motion blur via
        // the same Primitive::startTransform/endTransform interpolation
        // every other primitive already uses) - empty means static
        // instances (each expanded Primitive's endTransform ==
        // startTransform). A size mismatch is treated as "static", same
        // "malformed optional data degrades safely" convention as
        // Mesh::HasMotion().
        std::vector<Transform> instanceStart;
        std::vector<Transform> instanceEnd;
    };
}
