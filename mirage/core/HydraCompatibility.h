#pragma once

#include "mirage/core/Renderer.h"
#include "mirage/core/Scene.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace Mirage
{
    // Forward declarations
    class HydraRenderParam;
    class HydraRenderBuffer;

    // Enum for render task tokens (similar to Hydra's HdRenderTaskTokens)
    enum class RenderTaskToken
    {
        Camera,
        Viewport,
        Collection,
        RenderTags,
        EnableLighting,
        EnableSceneMaterials,
        EnableSceneLights
    };

    // Resource registry for tracking and managing resources
    class ResourceRegistry
    {
    public:
        ResourceRegistry() = default;
        ~ResourceRegistry() = default;

        // Register a resource with the registry
        template<typename T>
        void RegisterResource(const std::string& id, std::shared_ptr<T> resource)
        {
            resources[id] = resource;
        }

        // Get a resource from the registry
        template<typename T>
        std::shared_ptr<T> GetResource(const std::string& id)
        {
            auto it = resources.find(id);
            if (it != resources.end())
            {
                return std::static_pointer_cast<T>(it->second);
            }
            return nullptr;
        }

        // Remove a resource from the registry
        void RemoveResource(const std::string& id)
        {
            resources.erase(id);
        }

        // Clear all resources
        void Clear()
        {
            resources.clear();
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<void>> resources;
    };

    // Hydra compatible renderer interface
    class HydraCompatibleRenderer : public Renderer
    {
    public:
        HydraCompatibleRenderer(const Scene* s);
        virtual ~HydraCompatibleRenderer();

        // Implement the base Renderer interface
        virtual void Init(int width, int height) override;
        virtual void Render(const Camera& c, const Options& options, Color* output, AovBuffers* aovs = nullptr) override;

        // Hydra-compatible methods
        virtual void SetRenderParam(const std::string& paramName, const void* value);
        virtual void CommitResources();
        virtual void Sync();
        
        // Resource management
        ResourceRegistry& GetResourceRegistry() { return resourceRegistry; }

    protected:
        const Scene* scene;
        ResourceRegistry resourceRegistry;
        std::unique_ptr<Renderer> internalRenderer;
        
        // Hydra-compatible render settings
        std::unordered_map<std::string, void*> renderParams;
        bool resourcesDirty = true;
        bool needsSync = true;
    };

    // Create a Hydra-compatible renderer
    Renderer* CreateHydraCompatibleRenderer(const Scene* s);
}
