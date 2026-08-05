#pragma once

#include "mirage/utils/MathUtils.h"
#include "mirage/shaders/Material.h"
#include "mirage/prims/Mesh.h"

namespace Mirage
{
    enum Type
    {
        eSphere,
        ePlane,
        eMesh,
        // Finite-area punctual light shapes - unlike eSphere/ePlane/eMesh
        // these exist primarily to be sampled as area lights (Primitive::
        // lightSamples > 0 via a nonzero-emission material, same convention
        // as every other primitive type), though they're regular
        // intersectable/camera-visible geometry like any other primitive.
        // Point/directional (delta-distribution, non-area, non-visible)
        // lights are a separate concept - see Mirage::PunctualLight
        // (mirage/lights/PunctualLight.h) and Scene::lights, deliberately
        // outside this enum/union since they have no area or intersection
        // geometry to speak of.
        eRect,
        eDisk
    };

    struct SphereGeometry
    {
        float radius;
    };

    struct PlaneGeometry
    {
        float plane[4];
    };

    // Local-space rectangle in the primitive's XY plane (normal = local +Z),
    // centered at the origin, oriented/positioned via
    // Primitive::startTransform/endTransform - same "geometry data + shared
    // transform" convention as SphereGeometry/PlaneGeometry.
    struct RectGeometry
    {
        float width;
        float height;
    };

    // Local-space disk in the primitive's XY plane (normal = local +Z),
    // centered at the origin, oriented/positioned via
    // Primitive::startTransform/endTransform.
    struct DiskGeometry
    {
        float radius;
    };

    struct MeshGeometry
    {

        const Vec3 *positions;
        const Vec3 *normals;
        const int *indices;
        const Vec2 *uvs;
        const BVHNode *nodes;
        const float *cdf;

        // Deforming (2-keyframe) motion blur: end-of-shutter vertex snapshot,
        // parallel to `positions` (same index space) - nullptr for a static
        // mesh (Mesh::HasMotion() == false). See GeometryFromMesh.
        const Vec3 *positionsEnd;

        // Tangent-space normal mapping: per-vertex tangent, parallel to
        // `positions` (same index space) - nullptr when the mesh has no
        // tangent basis (Mesh::HasTangents() == false, e.g. no UV data or
        // ComputeTangents() was never called). See GeometryFromMesh.
        const Vec3 *tangents;

        int numVertices;
        int numIndices;
        int numNodes;

        float area;

        unsigned long id;

        // Index into Scene::meshes - lets VulkanRenderer::UploadScene resolve
        // this primitive's GpuMeshDescriptor in O(1) instead of the O(meshes)
        // pointer-identity scan against `id` above. -1 (set by
        // GeometryFromMesh, matching every other field in this struct) means
        // "unresolved" - UploadScene falls back to the `id`-based scan in
        // that case, so callers that predate this field (e.g. an
        // out-of-repo Hydra delegate) keep working unchanged. No in-class
        // default initializer here deliberately - MeshGeometry sits inside
        // Primitive's anonymous union, where a non-trivial member default
        // would make the union (and Primitive()) fail to default-construct.
        int32_t meshIndex;
    };

    struct Primitive
    {
        Primitive() : lightSamples(0) {}

        // begin end transforms for the primitive
        Transform startTransform;
        Transform endTransform;

        union
        {
            SphereGeometry sphere;
            PlaneGeometry plane;
            MeshGeometry mesh;
            RectGeometry rect;
            DiskGeometry disk;
        };

        Type type;

        // Index into Scene::materials (-1 = unassigned). Materials are shared/
        // deduplicated resources owned by Scene, not per-primitive values - see
        // Scene::AddMaterial/FindOrAddMaterial/GetMaterial.
        int32_t materialIndex = -1;

        // Stable identifier for the primId AOV, set by the scene owner (e.g.
        // hdMirage sets this to the Hydra rprim's HdRprim::GetPrimId() so the
        // AOV correlates with the host's own picking/selection IDs). -1 =
        // unset.
        int32_t hydraId = -1;

        // if > 0 then primitive will be explicitly sampled
        int lightSamples;
    };
}
