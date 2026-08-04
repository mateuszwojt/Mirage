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
        eMesh
    };

    struct SphereGeometry
    {
        float radius;
    };

    struct PlaneGeometry
    {
        float plane[4];
    };

    struct MeshGeometry
    {

        const Vec3 *positions;
        const Vec3 *normals;
        const int *indices;
        const Vec2 *uvs;
        const BVHNode *nodes;
        const float *cdf;

        int numVertices;
        int numIndices;
        int numNodes;

        float area;

        unsigned long id;
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
