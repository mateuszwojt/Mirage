#include "mirage/core/HydraCompatibility.h"
#include "mirage/utils/Util.h"

namespace Mirage
{
    HydraCompatibleRenderer::HydraCompatibleRenderer(const Scene* s)
        : scene(s)
    {
        // Create the internal renderer using CPU renderer
        // GPU renderer support can be added later when available
        internalRenderer.reset(CreateCpuRenderer(s));
    }

    HydraCompatibleRenderer::~HydraCompatibleRenderer()
    {
        // Clear all resources
        resourceRegistry.Clear();
        renderParams.clear();
    }

    void HydraCompatibleRenderer::Init(int width, int height)
    {
        if (internalRenderer) {
            internalRenderer->Init(width, height);
        }
    }

    void HydraCompatibleRenderer::Render(const Camera& c, const Options& options, Color* output, AovBuffers* aovs)
    {
        // Sync resources if needed
        if (needsSync) {
            Sync();
        }

        // Commit resources if dirty
        if (resourcesDirty) {
            CommitResources();
        }

        // Forward to internal renderer
        if (internalRenderer) {
            internalRenderer->Render(c, options, output, aovs);
        }
        else {
            // Fill with black if no renderer
            memset(output, 0, options.width * options.height * sizeof(Color));
        }
    }

    void HydraCompatibleRenderer::SetRenderParam(const std::string& paramName, const void* value)
    {
        if (value) {
            renderParams[paramName] = const_cast<void*>(value);
        }
        else {
            renderParams.erase(paramName);
        }
        
        // Mark as needing sync
        needsSync = true;
    }

    void HydraCompatibleRenderer::CommitResources()
    {
        // This would normally commit GPU resources, textures, etc.
        // For now, just mark as clean
        resourcesDirty = false;
    }

    void HydraCompatibleRenderer::Sync()
    {
        // Synchronize with the scene data
        // This would normally update internal data structures based on scene changes
        needsSync = false;
        resourcesDirty = true;
    }

    Renderer* CreateHydraCompatibleRenderer(const Scene* s)
    {
        return new HydraCompatibleRenderer(s);
    }
}
