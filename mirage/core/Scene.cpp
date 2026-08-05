#include "mirage/core/Scene.h"
#include "mirage/core/Intersection.h"
#include "mirage/utils/Util.h"

namespace Mirage
{
    void Scene::Clear()
    {
        meshes.clear();
        primitives.clear();
        instancers.clear();
        lights.clear();
        materials.clear();
        textures.clear();
        materialNameToIndex.clear();
        texturePathToIndex.clear();

        delete bvh.nodes;
        bvh.nodes = nullptr;

        sky = Sky();
    }

    void Scene::AddPrimitive(const Primitive &prim)
    {
        std::lock_guard<std::mutex> lock(mutex);
        primitives.push_back(prim);
    }

    void Scene::AddInstancer(const PointInstancer &instancer)
    {
        std::lock_guard<std::mutex> lock(mutex);
        instancers.push_back(instancer);
    }

    int Scene::AddMesh(std::unique_ptr<Mesh> mesh)
    {
        if (!mesh)
            return -1;

        std::lock_guard<std::mutex> lock(mutex);
        meshes.push_back(std::move(mesh));
        return (int)meshes.size() - 1;
    }

    int Scene::AddMaterial(std::unique_ptr<Material> mat)
    {
        std::lock_guard<std::mutex> lock(mutex);
        materials.push_back(std::move(mat));
        return (int)materials.size() - 1;
    }

    int Scene::FindOrAddMaterial(const std::string &name, std::unique_ptr<Material> mat)
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = materialNameToIndex.find(name);
        if (it != materialNameToIndex.end())
            return it->second;

        materials.push_back(std::move(mat));
        int index = (int)materials.size() - 1;
        materialNameToIndex[name] = index;
        return index;
    }

    Material *Scene::GetMaterial(int index) const
    {
        if (index < 0 || (size_t)index >= materials.size())
            return nullptr;
        return materials[index].get();
    }

    int Scene::AddTexture(std::unique_ptr<Texture> tex)
    {
        std::lock_guard<std::mutex> lock(mutex);
        textures.push_back(std::move(tex));
        return (int)textures.size() - 1;
    }

    int Scene::FindOrAddTexture(const std::string &path, std::unique_ptr<Texture> tex)
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = texturePathToIndex.find(path);
        if (it != texturePathToIndex.end())
            return it->second;

        textures.push_back(std::move(tex));
        int index = (int)textures.size() - 1;
        texturePathToIndex[path] = index;
        return index;
    }

    Texture *Scene::GetTexture(int index) const
    {
        if (index < 0 || (size_t)index >= textures.size())
            return nullptr;
        return textures[index].get();
    }

    void Scene::ExpandInstancers()
    {
        for (const PointInstancer &instancer : instancers)
        {
            // Resolve the shared mesh geometry once per instancer, not once
            // per instance - every expanded eMesh Primitive below gets an
            // identical MeshGeometry (same positions/indices/meshIndex),
            // exactly the "N primitives, one shared mesh" case
            // VulkanRenderer::UploadScene's meshIndex fast path exists for.
            MeshGeometry meshGeo{};
            if (instancer.type == eMesh)
            {
                if (instancer.meshIndex < 0 || (size_t)instancer.meshIndex >= meshes.size() || !meshes[instancer.meshIndex])
                    continue; // invalid mesh reference - skip this instancer entirely, not per-instance

                meshGeo = GeometryFromMesh(meshes[instancer.meshIndex].get());
                meshGeo.meshIndex = instancer.meshIndex;
            }

            const bool hasEndTransforms = !instancer.instanceEnd.empty() &&
                                           instancer.instanceEnd.size() == instancer.instanceStart.size();

            for (size_t i = 0; i < instancer.instanceStart.size(); ++i)
            {
                Primitive prim;
                prim.type = instancer.type;
                prim.startTransform = instancer.instanceStart[i];
                prim.endTransform = hasEndTransforms ? instancer.instanceEnd[i] : instancer.instanceStart[i];
                prim.materialIndex = instancer.materialIndex;
                prim.lightSamples = instancer.lightSamples;
                prim.hydraId = instancer.hydraId;

                switch (instancer.type)
                {
                case eSphere:
                    prim.sphere.radius = instancer.sphereRadius;
                    break;
                case ePlane:
                    prim.plane.plane[0] = instancer.plane[0];
                    prim.plane.plane[1] = instancer.plane[1];
                    prim.plane.plane[2] = instancer.plane[2];
                    prim.plane.plane[3] = instancer.plane[3];
                    break;
                case eMesh:
                    prim.mesh = meshGeo;
                    break;
                case eRect:
                    prim.rect.width = instancer.rectWidth;
                    prim.rect.height = instancer.rectHeight;
                    break;
                case eDisk:
                    prim.disk.radius = instancer.diskRadius;
                    break;
                }

                primitives.push_back(prim);
            }
        }

        // Instances are now ordinary primitives - clear so a second Build()
        // call (e.g. after adding more geometry) doesn't re-expand and
        // duplicate them.
        instancers.clear();
    }

    void Scene::Build()
    {
        std::lock_guard<std::mutex> lock(mutex);

        ExpandInstancers();

        if (primitives.size() == 0)
        {
            // nothing to build the scene from, should yield some error
            return;
        }

        // build scene bvh
        std::vector<Bounds> primitiveBounds;

        for (size_t i = 0; i < primitives.size(); ++i)
        {
            // Debug: Check primitive type and mesh data
            // if (primitives[i].type == eMesh) {
            //     printf("Processing mesh primitive %d: nodes=%p, numNodes=%d, positions=%p, numVertices=%d",
            //             (int)i, primitives[i].mesh.nodes, primitives[i].mesh.numNodes,
            //             primitives[i].mesh.positions, primitives[i].mesh.numVertices);
            // }
            
            Bounds r = PrimitiveBounds(primitives[i]);
            primitiveBounds.push_back(r);
        }

        BVHBuilder builder;
        bvh = builder.Build(&primitiveBounds[0], primitiveBounds.size());
    }
}