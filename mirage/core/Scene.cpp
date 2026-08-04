#include "mirage/core/Scene.h"
#include "mirage/core/Intersection.h"

namespace Mirage
{
    void Scene::Clear()
    {
        meshes.clear();
        primitives.clear();
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

    void Scene::Build()
    {
        std::lock_guard<std::mutex> lock(mutex);
        
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