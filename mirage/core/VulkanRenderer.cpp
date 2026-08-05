#include "mirage/core/VulkanRenderer.h"
#include "mirage/core/Scene.h"
#include "mirage/kernels/GpuLayout.h"
#include "mirage/utils/Util.h"

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace rhi;

namespace Mirage
{
    namespace
    {
        // Explicit field-by-field re-encoding, not a memcpy/reinterpret_cast of
        // BVHNode's bitfields - sidesteps the "C++ bitfield layout is
        // implementation-defined" risk entirely, since both the encoding (here)
        // and decoding (BVH.slang's IsLeaf()/RightIndex()) are ours to define.
        GpuBVHNode ConvertBVHNode(const BVHNode &n)
        {
            GpuBVHNode g{};
            g.lowerX = n.bounds.lower.x; g.lowerY = n.bounds.lower.y; g.lowerZ = n.bounds.lower.z;
            g.upperX = n.bounds.upper.x; g.upperY = n.bounds.upper.y; g.upperZ = n.bounds.upper.z;
            g.leftIndex = (uint32_t)n.leftIndex;
            g.rightIndexAndLeaf = ((uint32_t)n.rightIndex & 0x7FFFFFFFu) | (n.leaf ? 0x80000000u : 0u);
            return g;
        }

        GpuTransform ConvertTransform(const Transform &t)
        {
            GpuTransform g{};
            g.px = t.p.x; g.py = t.p.y; g.pz = t.p.z;
            g.rx = t.r.x; g.ry = t.r.y; g.rz = t.r.z; g.rw = t.r.w;
            g.s = t.s;
            return g;
        }

        GpuMaterial ConvertMaterial(const Material &m)
        {
            GpuMaterial g{};
            g.colorX = m.color.x; g.colorY = m.color.y; g.colorZ = m.color.z;
            g.emissionX = m.emission.x; g.emissionY = m.emission.y; g.emissionZ = m.emission.z;
            g.absorptionX = m.absorption.x; g.absorptionY = m.absorption.y; g.absorptionZ = m.absorption.z;
            g.eta = m.eta;
            g.metallic = m.metallic;
            g.subsurface = m.subsurface;
            g.specular = m.specular;
            g.roughness = m.roughness;
            g.specularTint = m.specularTint;
            g.anisotropic = m.anisotropic;
            g.sheen = m.sheen;
            g.sheenTint = m.sheenTint;
            g.clearcoat = m.clearcoat;
            g.clearcoatGloss = m.clearcoatGloss;
            g.transmission = m.transmission;
            g.opacity = m.opacity;
            g.albedoTextureIndex = m.albedoTextureIndex;
            g.roughnessTextureIndex = m.roughnessTextureIndex;
            g.metallicTextureIndex = m.metallicTextureIndex;
            g.opacityTextureIndex = m.opacityTextureIndex;
            g.normalTextureIndex = m.normalTextureIndex;
            return g;
        }

        GpuLight ConvertLight(const PunctualLight &l)
        {
            GpuLight g{};
            g.posX = l.position.x; g.posY = l.position.y; g.posZ = l.position.z;
            g.dirX = l.direction.x; g.dirY = l.direction.y; g.dirZ = l.direction.z;
            g.colorX = l.color.x; g.colorY = l.color.y; g.colorZ = l.color.z;
            g.intensity = l.intensity;
            g.radius = l.radius;
            g.angle = l.angle;
            g.type = (int32_t)l.type;
            return g;
        }
    }

    struct VulkanRenderer::Impl
    {
        // float3/float4 with no CUDA_CALLABLE/Vec3 dependency, matching Slang's
        // scalar-layout types (tightly packed floats) - avoids pulling
        // math/Vec3.h's full API surface into this translation unit just for a
        // POD upload buffer.
        struct float2_ { float x, y; };
        struct float3_ { float x, y, z; };
        struct float4_ { float x, y, z, w; };

        const Scene *scene;
        ComPtr<IDevice> device;
        ComPtr<IComputePipeline> clearColorPipeline;
        ComPtr<IComputePipeline> normalsPipeline;
        ComPtr<IComputePipeline> bsdfTestPipeline;
        ComPtr<IComputePipeline> pathTracePipeline;
        ComPtr<IComputePipeline> pathTraceTestPipeline;
        ComPtr<IComputePipeline> probeTestPipeline;
        // M5 validation-only: TextureArraySpike.slang dispatch. See RunTextureArraySpike().
        ComPtr<IComputePipeline> textureArraySpikePipeline;
        // M8 validation-only: TextureSampleTestMain.slang dispatch. See RunTextureSampleTest().
        ComPtr<IComputePipeline> textureSampleTestPipeline;
        bool available = false;
        int width = 0;
        int height = 0;

        // Scene buffers, uploaded once (Scene has no dirty-tracking today - see
        // the plan's "no existing change-tracking on Mirage::Scene" risk).
        bool sceneUploaded = false;
        ComPtr<IBuffer> primitivesBuf, meshDescBuf, sceneBvhBuf, meshBvhBuf, positionsBuf, normalsBuf, indicesBuf, uvsBuf;
        // Deforming (2-keyframe) motion blur (Tier-1 feature 2): end-of-shutter
        // vertex mega-buffer, parallel to positionsBuf - padded to one dummy
        // element (never indexed) when no mesh in the scene has motion, same
        // as every other CreateStructuredBuffer call below.
        ComPtr<IBuffer> positionsEndBuf;
        // Tangent-space normal mapping (Tier-1 feature 5): per-vertex
        // tangent mega-buffer, parallel to normalsBuf - padded to one dummy
        // element (never indexed) when no mesh in the scene has tangent
        // data, same as every other CreateStructuredBuffer call below.
        ComPtr<IBuffer> tangentsBuf;
        ComPtr<IBuffer> materialsBuf, skyBuf;
        // Point/directional punctual lights (Tier-2) - padded to one dummy
        // (never-indexed, since numLightsUploaded stays 0) element when the
        // scene has no lights, same convention as positionsEndBuf/tangentsBuf.
        ComPtr<IBuffer> lightsBuf;
        uint32_t numPrimitivesUploaded = 0;
        uint32_t numLightsUploaded = 0;

        // Per-material textures (albedo/roughness/metallic maps, M8) - fixed-
        // size array of kMaxMaterialTextures slots (Material.slang/
        // Constants.slang's g_Textures[]/kMaxMaterialTextures), confirmed
        // viable by the M5 spike. Real textures for scene->textures, a shared
        // 1x1 white dummy for the remaining slots - same "always bind
        // something, the pipeline's descriptor layout is fixed regardless of
        // runtime scene content" reasoning as probeTexture's dummy.
        static const uint32_t kMaxMaterialTextures = 32;
        std::vector<ComPtr<ITexture>> materialTextures;
        ComPtr<ISampler> materialTextureSampler;
        // Per-slot width/height/UDIM-grid metadata (Tier-2 "mipmapping"/
        // "UDIM"), built alongside materialTextures by UploadMaterialTextures()
        // and uploaded as textureInfoBuf by UploadScene() - see
        // UploadMaterialTextures()'s comment.
        std::vector<GpuTextureInfo> textureInfos;
        ComPtr<IBuffer> textureInfoBuf;

        // HDR environment probe (sampling/Probe.slang). Always populated -
        // with a 1x1 black dummy when the scene has no valid probe - since
        // Sky_Eval()/PathTrace() statically reference these bindings regardless
        // of the runtime probeValid branch (the pipeline's descriptor layout is
        // fixed at link time; only the *code path taken* is conditional).
        ComPtr<ITexture> probeTexture;
        ComPtr<ISampler> probeSampler;
        ComPtr<IBuffer> probeDataBuf, probeCdfXBuf, probePdfXBuf, probeCdfYBuf, probePdfYBuf;
        int probeWidth = 1, probeHeight = 1;

        // Progressive accumulation state for RenderMode::ePathTrace (see
        // AddSample.slang / PathTraceMain.slang). Reallocated/zeroed whenever
        // the requested output size changes.
        ComPtr<IBuffer> accumBuffer;
        int accumWidth = 0, accumHeight = 0;
        uint32_t accumulatedSamples = 0;

        // First-hit AOV buffers (depth/normal/primId) - same size/lifetime as
        // accumBuffer, reallocated together in AllocateAccumBuffer(). Unlike
        // accumBuffer these hold a single-valued snapshot, not a progressive
        // sum, but they still need per-dimension (re)allocation the same way.
        ComPtr<IBuffer> depthBuffer, normalBuffer, primIdBuffer;

        bool CreateDevice()
        {
            DeviceDesc desc;
            desc.deviceType = DeviceType::Vulkan;

            const char *searchPaths[] = MIRAGE_SLANG_SEARCH_PATHS;
            desc.slang.searchPaths = searchPaths;
            desc.slang.searchPathCount = (uint32_t)(sizeof(searchPaths) / sizeof(searchPaths[0]));

            Result result = getRHI()->createDevice(desc, device.writeRef());
            if (SLANG_FAILED(result) || !device)
            {
                fprintf(stderr, "VulkanRenderer: failed to create Vulkan device via slang-rhi (result=0x%08x). "
                                "Falling back to CPU renderer.\n", (unsigned)result);
                return false;
            }
            return true;
        }

        // Compiles `moduleName`'s "main" entry point (loaded via the device's
        // Slang session, resolved against MIRAGE_SLANG_SEARCH_PATHS) into a
        // compute pipeline. See VulkanRenderer.h for why this is session-based
        // rather than loading precompiled SPIR-V directly.
        bool CreatePipeline(const char *moduleName, ComPtr<IComputePipeline> &outPipeline)
        {
            ComPtr<slang::ISession> session = device->getSlangSession();
            if (!session)
            {
                fprintf(stderr, "VulkanRenderer: device has no Slang session\n");
                return false;
            }

            ComPtr<slang::IBlob> diagnostics;
            slang::IModule *module = session->loadModule(moduleName, diagnostics.writeRef());
            if (diagnostics && diagnostics->getBufferSize() > 0)
                fprintf(stderr, "VulkanRenderer (slang, %s): %s\n", moduleName, (const char *)diagnostics->getBufferPointer());
            if (!module)
                return false;

            ComPtr<slang::IEntryPoint> entryPoint;
            if (SLANG_FAILED(module->findEntryPointByName("main", entryPoint.writeRef())) || !entryPoint)
            {
                fprintf(stderr, "VulkanRenderer: %s.slang has no 'main' entry point\n", moduleName);
                return false;
            }

            std::vector<slang::IComponentType *> components = {module, entryPoint.get()};
            ComPtr<slang::IComponentType> composed;
            diagnostics = nullptr;
            if (SLANG_FAILED(session->createCompositeComponentType(
                    components.data(), components.size(), composed.writeRef(), diagnostics.writeRef())))
            {
                if (diagnostics)
                    fprintf(stderr, "VulkanRenderer (slang, %s): %s\n", moduleName, (const char *)diagnostics->getBufferPointer());
                return false;
            }

            ComPtr<slang::IComponentType> linked;
            diagnostics = nullptr;
            if (SLANG_FAILED(composed->link(linked.writeRef(), diagnostics.writeRef())))
            {
                if (diagnostics)
                    fprintf(stderr, "VulkanRenderer (slang, %s): %s\n", moduleName, (const char *)diagnostics->getBufferPointer());
                return false;
            }

            ShaderProgramDesc programDesc = {};
            programDesc.slangGlobalScope = linked.get();

            ComPtr<IShaderProgram> program;
            ComPtr<ISlangBlob> programDiagnostics;
            if (SLANG_FAILED(device->createShaderProgram(programDesc, program.writeRef(), programDiagnostics.writeRef())))
            {
                if (programDiagnostics)
                    fprintf(stderr, "VulkanRenderer (rhi, %s): %s\n", moduleName, (const char *)programDiagnostics->getBufferPointer());
                return false;
            }

            ComputePipelineDesc pipelineDesc = {};
            pipelineDesc.program = program.get();
            if (SLANG_FAILED(device->createComputePipeline(pipelineDesc, outPipeline.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create compute pipeline for %s\n", moduleName);
                return false;
            }
            return true;
        }

        template <typename T>
        ComPtr<IBuffer> CreateStructuredBuffer(const std::vector<T> &data)
        {
            BufferDesc desc = {};
            desc.size = data.empty() ? sizeof(T) : data.size() * sizeof(T);
            desc.elementSize = sizeof(T);
            desc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            desc.defaultState = ResourceState::ShaderResource;
            desc.memoryType = MemoryType::DeviceLocal;

            // slang-rhi requires non-null initial data when a size is specified;
            // pad empty arrays to one zeroed element (never indexed by the shader
            // when the corresponding count is 0).
            std::vector<T> padded;
            const T *src = data.data();
            if (data.empty())
            {
                padded.resize(1);
                src = padded.data();
            }

            ComPtr<IBuffer> buffer;
            device->createBuffer(desc, src, buffer.writeRef());
            return buffer;
        }

        // Uploads the HDR environment probe (sampling/Probe.slang), or a 1x1
        // black dummy if the scene has none - see the member comment on
        // probeTexture for why a dummy is required even when unused.
        void UploadProbe()
        {
            const Probe &probe = scene->sky.probe;

            if (probe.valid)
            {
                probeWidth = probe.width;
                probeHeight = probe.height;
                size_t pixelCount = (size_t)probeWidth * (size_t)probeHeight;

                std::vector<float4_> texels(pixelCount);
                for (size_t i = 0; i < pixelCount; ++i)
                    texels[i] = {probe.data[i].x, probe.data[i].y, probe.data[i].z, probe.data[i].w};

                TextureDesc texDesc = {};
                texDesc.type = TextureType::Texture2D;
                texDesc.format = Format::RGBA32Float;
                texDesc.size = {(uint32_t)probeWidth, (uint32_t)probeHeight, 1};
                texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
                texDesc.memoryType = MemoryType::DeviceLocal;

                SubresourceData sub = {};
                sub.data = texels.data();
                sub.rowPitch = (Size)probeWidth * sizeof(float4_);
                sub.slicePitch = 0;
                device->createTexture(texDesc, &sub, probeTexture.writeRef());

                std::vector<float4_> probeData(texels); // g_ProbeData: exact per-texel fetch for ProbeSample
                std::vector<float> cdfX(probe.cdfValuesX, probe.cdfValuesX + pixelCount);
                std::vector<float> pdfX(probe.pdfValuesX, probe.pdfValuesX + pixelCount);
                std::vector<float> cdfY(probe.cdfValuesY, probe.cdfValuesY + probeHeight);
                std::vector<float> pdfY(probe.pdfValuesY, probe.pdfValuesY + probeHeight);

                probeDataBuf = CreateStructuredBuffer(probeData);
                probeCdfXBuf = CreateStructuredBuffer(cdfX);
                probePdfXBuf = CreateStructuredBuffer(pdfX);
                probeCdfYBuf = CreateStructuredBuffer(cdfY);
                probePdfYBuf = CreateStructuredBuffer(pdfY);
            }
            else
            {
                probeWidth = 1;
                probeHeight = 1;

                std::vector<float4_> black{{0.0f, 0.0f, 0.0f, 0.0f}};
                TextureDesc texDesc = {};
                texDesc.type = TextureType::Texture2D;
                texDesc.format = Format::RGBA32Float;
                texDesc.size = {1, 1, 1};
                texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
                texDesc.memoryType = MemoryType::DeviceLocal;
                SubresourceData sub = {};
                sub.data = black.data();
                sub.rowPitch = sizeof(float4_);
                sub.slicePitch = 0;
                device->createTexture(texDesc, &sub, probeTexture.writeRef());

                std::vector<float4_> dummyData{{0.0f, 0.0f, 0.0f, 0.0f}};
                std::vector<float> zero1 = {0.0f};
                probeDataBuf = CreateStructuredBuffer(dummyData);
                probeCdfXBuf = CreateStructuredBuffer(zero1);
                probePdfXBuf = CreateStructuredBuffer(zero1);
                probeCdfYBuf = CreateStructuredBuffer(zero1);
                probePdfYBuf = CreateStructuredBuffer(zero1);
            }

            SamplerDesc samplerDesc = {};
            samplerDesc.minFilter = TextureFilteringMode::Linear;
            samplerDesc.magFilter = TextureFilteringMode::Linear;
            samplerDesc.addressU = TextureAddressingMode::Wrap;
            samplerDesc.addressV = TextureAddressingMode::ClampToEdge;
            samplerDesc.addressW = TextureAddressingMode::ClampToEdge;
            device->createSampler(samplerDesc, probeSampler.writeRef());
        }

        // Uploads the per-material texture array (Material.slang's
        // g_Textures[kMaxMaterialTextures]) - real textures for
        // scene->textures, a shared 1x1 opaque-white dummy for the remaining
        // slots. Always fills every slot (never leaves one unbound), for the
        // same reason UploadProbe() always binds *something*: the compute
        // pipeline's descriptor layout is fixed at link time regardless of how
        // many textures the current scene actually has.
        //
        // Also builds textureInfos (Tier-2 "mipmapping"/"UDIM") in lockstep,
        // one GpuTextureInfo per slot - PathTrace.slang's mip-level heuristic
        // needs a texture's base width/height to convert a world-space
        // footprint into a mip level (SampleLevel() itself needs no
        // dimensions, but choosing *which* level to ask for does), and
        // Material_Sample*'s UDIM remap needs the grid dimensions. See
        // UploadScene() for where textureInfosBuf is actually created (kept
        // there, not here, since every other scene-level buffer is created
        // together at the end of UploadScene() - this function only builds
        // the CPU-side vector).
        void UploadMaterialTextures()
        {
            materialTextures.clear();
            materialTextures.resize(kMaxMaterialTextures);
            textureInfos.clear();
            textureInfos.resize(kMaxMaterialTextures);

            SamplerDesc samplerDesc = {};
            samplerDesc.minFilter = TextureFilteringMode::Linear;
            samplerDesc.magFilter = TextureFilteringMode::Linear;
            samplerDesc.mipFilter = TextureFilteringMode::Linear;
            samplerDesc.addressU = TextureAddressingMode::Wrap;
            samplerDesc.addressV = TextureAddressingMode::Wrap;
            samplerDesc.addressW = TextureAddressingMode::Wrap;
            device->createSampler(samplerDesc, materialTextureSampler.writeRef());

            const size_t realCount = scene ? scene->textures.size() : 0;
            if (realCount > kMaxMaterialTextures)
            {
                // Slots >= kMaxMaterialTextures silently fall back to the
                // dummy white texture below with no error today - UDIM
                // atlases make hitting this ceiling considerably more
                // likely than before (one atlas per UDIM material, vs. one
                // slot per ordinary texture), so surface it explicitly
                // rather than let materials quietly render wrong.
                fprintf(stderr, "VulkanRenderer: scene has %zu textures, exceeding kMaxMaterialTextures (%u) - "
                                 "textures at index >= %u will render as opaque white on the GPU backend.\n",
                        realCount, kMaxMaterialTextures, kMaxMaterialTextures);
            }

            for (uint32_t i = 0; i < kMaxMaterialTextures; ++i)
            {
                TextureDesc texDesc = {};
                texDesc.type = TextureType::Texture2D;
                texDesc.format = Format::RGBA32Float;
                texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
                texDesc.memoryType = MemoryType::DeviceLocal;

                const Texture *tex = (i < realCount) ? scene->textures[i].get() : nullptr;
                if (tex && tex->data && tex->width > 0 && tex->height > 0)
                {
                    texDesc.size = {(uint32_t)tex->width, (uint32_t)tex->height, 1};

                    GpuTextureInfo info{};
                    info.width = tex->width;
                    info.height = tex->height;
                    info.udimGridWidth = tex->udimGridWidth;
                    info.udimGridHeight = tex->udimGridHeight;
                    textureInfos[i] = info;

                    if (tex->mipCount > 1 && tex->mipData)
                    {
                        // Full mip chain: one subresource per level, ordered
                        // mip 0..mipCount-1 (single array layer, so
                        // subresource index == mip index directly - see
                        // TextureDesc::getSubresourceCount()'s doc comment).
                        texDesc.mipCount = (uint32_t)tex->mipCount;

                        std::vector<SubresourceData> subs((size_t)tex->mipCount);
                        for (int level = 0; level < tex->mipCount; ++level)
                        {
                            subs[level].data = tex->mipData[level];
                            subs[level].rowPitch = (Size)tex->mipWidths[level] * sizeof(float4_);
                            subs[level].slicePitch = 0;
                        }
                        device->createTexture(texDesc, subs.data(), materialTextures[i].writeRef());
                    }
                    else
                    {
                        // No mip chain generated (e.g. GenerateMips() was
                        // skipped, or the texture predates this feature) -
                        // single level, exactly the pre-mipmapping behavior.
                        texDesc.mipCount = 1;

                        SubresourceData sub = {};
                        sub.data = tex->data;
                        sub.rowPitch = (Size)tex->width * sizeof(float4_);
                        sub.slicePitch = 0;
                        device->createTexture(texDesc, &sub, materialTextures[i].writeRef());
                    }
                }
                else
                {
                    std::vector<float4_> white{{1.0f, 1.0f, 1.0f, 1.0f}};
                    texDesc.size = {1, 1, 1};
                    texDesc.mipCount = 1;

                    SubresourceData sub = {};
                    sub.data = white.data();
                    sub.rowPitch = sizeof(float4_);
                    sub.slicePitch = 0;
                    device->createTexture(texDesc, &sub, materialTextures[i].writeRef());

                    GpuTextureInfo info{};
                    info.width = 1;
                    info.height = 1;
                    info.udimGridWidth = 1;
                    info.udimGridHeight = 1;
                    textureInfos[i] = info;
                }
            }
        }

        // Binds every slot of the per-material texture array via
        // cursor["g_Textures"][i] - the array-element indexing mechanism
        // confirmed working by the M5 spike (RunTextureArraySpike).
        void BindMaterialTextures(ShaderCursor &cursor)
        {
            ShaderCursor texturesCursor = cursor["g_Textures"];
            for (uint32_t i = 0; i < kMaxMaterialTextures; ++i)
                texturesCursor[i].setBinding(Binding(materialTextures[i], materialTextureSampler));
        }

        // Builds GPU buffers from *scene: primitives, mesh descriptors, scene BVH,
        // mesh BVH nodes (concatenated across meshes), positions/normals/indices
        // (concatenated, mesh-local indices - see GpuMeshDescriptor's offsets).
        // Mirrors what GpuRenderer's CUDA constructor used to do, minus the
        // per-mesh texture-object plumbing (replaced by mega-buffer offsets).
        void UploadScene()
        {
            if (sceneUploaded || !scene)
                return;

            // Map each Mesh* to its index in scene->meshes, so primitives of type
            // eMesh can resolve prim.mesh.id (set by GeometryFromMesh) to a
            // GpuMeshDescriptor index.
            std::vector<const Mesh *> meshPtrs;
            meshPtrs.reserve(scene->meshes.size());
            for (auto &m : scene->meshes)
                meshPtrs.push_back(m.get());

            std::vector<GpuMeshDescriptor> meshDescs;
            std::vector<float3_> positions, normals; // defined below as plain float3
            std::vector<float3_> positionsEnd; // deforming motion blur, see GpuMeshDescriptor::hasMotion
            std::vector<float3_> tangents; // normal mapping, see GpuMeshDescriptor::hasTangents
            std::vector<float2_> uvs;
            std::vector<int32_t> indices;
            std::vector<GpuBVHNode> meshBvhNodes;

            for (const Mesh *mesh : meshPtrs)
            {
                GpuMeshDescriptor desc{};
                desc.vertexOffset = (uint32_t)positions.size();
                desc.indexOffset = (uint32_t)indices.size();
                desc.uvOffset = (uint32_t)uvs.size();
                desc.hasUVs = mesh->uvs.empty() ? 0u : 1u;
                desc.nodeOffset = (uint32_t)meshBvhNodes.size();
                desc.numVertices = (uint32_t)mesh->vertices.size();
                desc.numIndices = (uint32_t)mesh->indices.size();
                desc.numNodes = (uint32_t)mesh->bvh.numNodes;
                desc.area = mesh->area;
                desc.hasMotion = mesh->HasMotion() ? 1u : 0u;
                desc.vertexOffsetEnd = (uint32_t)positionsEnd.size();
                desc.hasTangents = mesh->HasTangents() ? 1u : 0u;
                desc.tangentOffset = (uint32_t)tangents.size();
                meshDescs.push_back(desc);

                for (auto &v : mesh->vertices)
                    positions.push_back({v.x, v.y, v.z});
                for (auto &n : mesh->normals)
                    normals.push_back({n.x, n.y, n.z});
                for (auto &uv : mesh->uvs)
                    uvs.push_back({uv.x, uv.y});
                for (int idx : mesh->indices)
                    indices.push_back(idx);
                for (int i = 0; i < mesh->bvh.numNodes; ++i)
                    meshBvhNodes.push_back(ConvertBVHNode(mesh->bvh.nodes[i]));
                if (mesh->HasMotion())
                {
                    for (auto &v : mesh->verticesEnd)
                        positionsEnd.push_back({v.x, v.y, v.z});
                }
                if (mesh->HasTangents())
                {
                    for (auto &t : mesh->tangents)
                        tangents.push_back({t.x, t.y, t.z});
                }
            }

            // One GpuMaterial per Scene::materials entry (shared/deduplicated -
            // Scene::FindOrAddMaterial is what actually collapses duplicates;
            // this just uploads whatever the registry ended up holding), plus
            // one trailing default-constructed entry as a safe fallback target
            // for any primitive with an unassigned/out-of-range materialIndex
            // (a Primitive::materialIndex of -1 is valid on the CPU path, which
            // falls back to Material() via ResolveMaterial() in Renderer.cpp -
            // the GPU has no such per-access fallback, so invalid indices must
            // be resolved to a concrete, in-bounds slot at upload time instead).
            std::vector<GpuMaterial> materials;
            materials.reserve(scene->materials.size() + 1);
            for (const auto &mat : scene->materials)
                materials.push_back(ConvertMaterial(mat ? *mat : Material()));
            const int32_t defaultMaterialIndex = (int32_t)materials.size();
            materials.push_back(ConvertMaterial(Material()));

            std::vector<GpuPrimitive> primitives;
            primitives.reserve(scene->primitives.size());
            for (const Primitive &prim : scene->primitives)
            {
                GpuPrimitive gp{};
                gp.startTransform = ConvertTransform(prim.startTransform);
                gp.endTransform = ConvertTransform(prim.endTransform);
                gp.type = (int32_t)prim.type;
                gp.lightSamples = prim.lightSamples;
                gp.meshIndex = -1;
                gp.materialIndex = (prim.materialIndex >= 0 && (size_t)prim.materialIndex < scene->materials.size())
                                        ? prim.materialIndex
                                        : defaultMaterialIndex;
                gp.hydraId = prim.hydraId;

                switch (prim.type)
                {
                case eSphere:
                    gp.unionDataX = prim.sphere.radius;
                    gp.unionDataY = gp.unionDataZ = gp.unionDataW = 0.0f;
                    break;
                case ePlane:
                    gp.unionDataX = prim.plane.plane[0];
                    gp.unionDataY = prim.plane.plane[1];
                    gp.unionDataZ = prim.plane.plane[2];
                    gp.unionDataW = prim.plane.plane[3];
                    break;
                case eMesh:
                    if (prim.mesh.meshIndex >= 0 && (size_t)prim.mesh.meshIndex < meshPtrs.size())
                    {
                        // Fast path: the scene owner already resolved this
                        // primitive's mesh index (e.g. via Scene::AddMesh's
                        // return value) - see MeshGeometry::meshIndex.
                        gp.meshIndex = prim.mesh.meshIndex;
                    }
                    else
                    {
                        // Fallback for callers that predate meshIndex (e.g.
                        // an out-of-repo Hydra delegate still using the
                        // original GeometryFromMesh()+AddMesh() pattern) -
                        // resolve by pointer identity instead, exactly as
                        // this loop always did before that field existed.
                        for (size_t i = 0; i < meshPtrs.size(); ++i)
                        {
                            if ((unsigned long)meshPtrs[i] == prim.mesh.id)
                            {
                                gp.meshIndex = (int32_t)i;
                                break;
                            }
                        }
                    }
                    break;
                case eRect:
                    gp.unionDataX = prim.rect.width;
                    gp.unionDataY = prim.rect.height;
                    gp.unionDataZ = gp.unionDataW = 0.0f;
                    break;
                case eDisk:
                    gp.unionDataX = prim.disk.radius;
                    gp.unionDataY = gp.unionDataZ = gp.unionDataW = 0.0f;
                    break;
                }
                primitives.push_back(gp);
            }

            UploadProbe();
            UploadMaterialTextures();
            textureInfoBuf = CreateStructuredBuffer(textureInfos);

            GpuSky gsky{};
            gsky.horizonX = scene->sky.horizon.x; gsky.horizonY = scene->sky.horizon.y; gsky.horizonZ = scene->sky.horizon.z;
            gsky.zenithX = scene->sky.zenith.x; gsky.zenithY = scene->sky.zenith.y; gsky.zenithZ = scene->sky.zenith.z;
            gsky.probeValid = scene->sky.probe.valid ? 1 : 0;
            gsky.probeWidth = probeWidth;
            gsky.probeHeight = probeHeight;
            std::vector<GpuSky> skyData{gsky};

            std::vector<GpuBVHNode> sceneBvhNodes;
            sceneBvhNodes.reserve(scene->bvh.numNodes);
            for (int i = 0; i < scene->bvh.numNodes; ++i)
                sceneBvhNodes.push_back(ConvertBVHNode(scene->bvh.nodes[i]));

            std::vector<GpuLight> lights;
            lights.reserve(scene->lights.size());
            for (const PunctualLight &light : scene->lights)
                lights.push_back(ConvertLight(light));
            numLightsUploaded = (uint32_t)lights.size();
            if (lights.empty())
                lights.push_back(GpuLight{}); // dummy - never indexed, numLightsUploaded stays 0

            primitivesBuf = CreateStructuredBuffer(primitives);
            meshDescBuf = CreateStructuredBuffer(meshDescs);
            sceneBvhBuf = CreateStructuredBuffer(sceneBvhNodes);
            meshBvhBuf = CreateStructuredBuffer(meshBvhNodes);
            positionsBuf = CreateStructuredBuffer(positions);
            positionsEndBuf = CreateStructuredBuffer(positionsEnd);
            tangentsBuf = CreateStructuredBuffer(tangents);
            normalsBuf = CreateStructuredBuffer(normals);
            indicesBuf = CreateStructuredBuffer(indices);
            uvsBuf = CreateStructuredBuffer(uvs);
            materialsBuf = CreateStructuredBuffer(materials);
            skyBuf = CreateStructuredBuffer(skyData);
            lightsBuf = CreateStructuredBuffer(lights);

            numPrimitivesUploaded = (uint32_t)primitives.size();
            sceneUploaded = true;
        }

        GpuCamera BuildGpuCamera(const Camera &camera, const Options &options)
        {
            float effectiveAperture = camera.EffectiveApertureDiameter();
            CameraSampler sampler(
                Transform(camera.position, camera.rotation),
                camera.EffectiveFov(), 0.001f, 1.0f,
                effectiveAperture, camera.focalPoint,
                options.enableDOF, options.width, options.height);

            GpuCamera gcam{};
            memcpy(gcam.rasterToWorld, sampler.rasterToWorld.cols, sizeof(gcam.rasterToWorld));
            Vec4 originCol = sampler.cameraToWorld.GetCol(3);
            gcam.originX = originCol.x; gcam.originY = originCol.y; gcam.originZ = originCol.z;
            gcam.aperture = effectiveAperture;
            gcam.focalPoint = camera.focalPoint;
            gcam.enableDOF = options.enableDOF ? 1u : 0u;
            gcam.shutterStart = camera.shutterStart;
            gcam.shutterEnd = camera.shutterEnd;
            gcam.fovY = camera.EffectiveFov();
            return gcam;
        }

        void RunNormals(const Camera &camera, const Options &options, Color *output)
        {
            UploadScene();

            GpuCamera gcam = BuildGpuCamera(camera, options);

            GpuFrameParams frame{};
            frame.width = (uint32_t)options.width;
            frame.height = (uint32_t)options.height;
            frame.numPrimitives = numPrimitivesUploaded;

            std::vector<GpuCamera> camData{gcam};
            std::vector<GpuFrameParams> frameData{frame};
            ComPtr<IBuffer> cameraBuf = CreateStructuredBuffer(camData);
            ComPtr<IBuffer> frameBuf = CreateStructuredBuffer(frameData);

            BufferDesc outDesc = {};
            outDesc.size = (size_t)options.width * (size_t)options.height * sizeof(float) * 4;
            outDesc.elementSize = sizeof(float) * 4;
            outDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                             BufferUsage::CopyDestination | BufferUsage::CopySource;
            outDesc.defaultState = ResourceState::UnorderedAccess;
            outDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> outputBuffer;
            if (SLANG_FAILED(device->createBuffer(outDesc, nullptr, outputBuffer.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create normals output buffer\n");
                return;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(normalsPipeline);

                ShaderCursor cursor(rootObject);
                cursor["g_Frame"].setBinding(frameBuf);
                cursor["g_Camera"].setBinding(cameraBuf);
                cursor["g_Output"].setBinding(outputBuffer);
                cursor["g_Primitives"].setBinding(primitivesBuf);
                cursor["g_MeshDescriptors"].setBinding(meshDescBuf);
                cursor["g_SceneBVH"].setBinding(sceneBvhBuf);
                cursor["g_MeshBVHNodes"].setBinding(meshBvhBuf);
                cursor["g_Positions"].setBinding(positionsBuf);
                cursor["g_PositionsEnd"].setBinding(positionsEndBuf);
                cursor["g_Tangents"].setBinding(tangentsBuf);
                cursor["g_Normals"].setBinding(normalsBuf);
                cursor["g_Indices"].setBinding(indicesBuf);

                uint32_t groupsX = ((uint32_t)options.width + 7) / 8;
                uint32_t groupsY = ((uint32_t)options.height + 7) / 8;
                passEncoder->dispatchCompute(groupsX, groupsY, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(outputBuffer, 0, outDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back normals output buffer\n");
                return;
            }
            memcpy(output, readback->getBufferPointer(), outDesc.size);
        }

        void AllocateAccumBuffer(int w, int h)
        {
            BufferDesc desc = {};
            desc.size = (size_t)w * (size_t)h * sizeof(float) * 4;
            desc.elementSize = sizeof(float) * 4;
            desc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                         BufferUsage::CopyDestination | BufferUsage::CopySource;
            desc.defaultState = ResourceState::UnorderedAccess;
            desc.memoryType = MemoryType::DeviceLocal;

            std::vector<float> zeros((size_t)w * (size_t)h * 4, 0.0f);
            device->createBuffer(desc, zeros.data(), accumBuffer.writeRef());
            device->createBuffer(desc, zeros.data(), depthBuffer.writeRef());
            device->createBuffer(desc, zeros.data(), normalBuffer.writeRef());
            device->createBuffer(desc, zeros.data(), primIdBuffer.writeRef());
            accumWidth = w;
            accumHeight = h;
            accumulatedSamples = 0;
        }

        void EnsureAccumBuffer(int w, int h)
        {
            if (accumBuffer && accumWidth == w && accumHeight == h)
                return;
            AllocateAccumBuffer(w, h);
        }

        // Re-zero the existing accumulation buffer in place (same size, no
        // scene/geometry touched) - used when only the camera moved, so
        // upcoming samples don't blend with pixels traced from the previous
        // camera position.
        void ResetAccumulation()
        {
            if (accumWidth > 0 && accumHeight > 0)
                AllocateAccumBuffer(accumWidth, accumHeight);
        }

        // Real per-pixel path-trace dispatch (RenderMode::ePathTrace). Each call
        // is one more accumulated sample (progressive box-filter accumulation,
        // see AddSample.slang); resolves by dividing by accumulated weight into
        // `output` every call, so a single call already produces a usable
        // (noisy, 1-spp-so-far) image. Per-pixel RNG is independent of the CPU
        // renderer's - see PathTraceMain.slang's header comment - so this is not
        // held to a strict CPU-vs-GPU pixel match; RunPathTraceTest() below is.
        void RunPathTrace(const Camera &camera, const Options &options, Color *output, AovBuffers *aovs = nullptr)
        {
            UploadScene();
            EnsureAccumBuffer(options.width, options.height);

            GpuCamera gcam = BuildGpuCamera(camera, options);

            GpuFrameParams frame{};
            frame.width = (uint32_t)options.width;
            frame.height = (uint32_t)options.height;
            frame.numPrimitives = numPrimitivesUploaded;
            frame.maxDepth = options.maxDepth;
            frame.frameSeed = accumulatedSamples * 9781u + 1u;
            frame.sampleIndex = accumulatedSamples;
            frame.clampValue = options.clamp;
            frame.numLights = numLightsUploaded;

            std::vector<GpuCamera> camData{gcam};
            std::vector<GpuFrameParams> frameData{frame};
            ComPtr<IBuffer> cameraBuf = CreateStructuredBuffer(camData);
            ComPtr<IBuffer> frameBuf = CreateStructuredBuffer(frameData);

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(pathTracePipeline);

                ShaderCursor cursor(rootObject);
                cursor["g_Frame"].setBinding(frameBuf);
                cursor["g_Camera"].setBinding(cameraBuf);
                cursor["g_Primitives"].setBinding(primitivesBuf);
                cursor["g_MeshDescriptors"].setBinding(meshDescBuf);
                cursor["g_SceneBVH"].setBinding(sceneBvhBuf);
                cursor["g_MeshBVHNodes"].setBinding(meshBvhBuf);
                cursor["g_Positions"].setBinding(positionsBuf);
                cursor["g_PositionsEnd"].setBinding(positionsEndBuf);
                cursor["g_Tangents"].setBinding(tangentsBuf);
                cursor["g_Normals"].setBinding(normalsBuf);
                cursor["g_Indices"].setBinding(indicesBuf);
                cursor["g_UVs"].setBinding(uvsBuf);
                cursor["g_Materials"].setBinding(materialsBuf);
                BindMaterialTextures(cursor);
                cursor["g_TextureInfo"].setBinding(textureInfoBuf);
                cursor["g_Sky"].setBinding(skyBuf);
                cursor["g_Lights"].setBinding(lightsBuf);
                cursor["g_ProbeTexture"].setBinding(Binding(probeTexture, probeSampler));
                cursor["g_ProbeData"].setBinding(probeDataBuf);
                cursor["g_ProbeCdfX"].setBinding(probeCdfXBuf);
                cursor["g_ProbePdfX"].setBinding(probePdfXBuf);
                cursor["g_ProbeCdfY"].setBinding(probeCdfYBuf);
                cursor["g_ProbePdfY"].setBinding(probePdfYBuf);
                cursor["g_AccumBuffer"].setBinding(accumBuffer);
                cursor["g_AovDepth"].setBinding(depthBuffer);
                cursor["g_AovNormal"].setBinding(normalBuffer);
                cursor["g_AovPrimId"].setBinding(primIdBuffer);

                uint32_t groupsX = ((uint32_t)options.width + 7) / 8;
                uint32_t groupsY = ((uint32_t)options.height + 7) / 8;
                passEncoder->dispatchCompute(groupsX, groupsY, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }
            ++accumulatedSamples;

            ComPtr<ISlangBlob> readback;
            size_t accumSize = (size_t)options.width * (size_t)options.height * sizeof(float) * 4;
            if (SLANG_FAILED(device->readBuffer(accumBuffer, 0, accumSize, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back accumulation buffer\n");
                return;
            }

            const float *acc = (const float *)readback->getBufferPointer();
            for (int i = 0; i < options.width * options.height; ++i)
            {
                float w = acc[i * 4 + 3];
                float invW = w > 0.0f ? 1.0f / w : 0.0f;
                output[i] = Color(acc[i * 4 + 0] * invW, acc[i * 4 + 1] * invW, acc[i * 4 + 2] * invW, w);
            }

            // AOV readback - gated on aovMask/non-null output pointer, unlike
            // the always-bound kernel writes above. This is the actual
            // bandwidth-saving point: the caller (HdMirageRenderer) only sets
            // aovMask on the first sample of a fresh accumulation window, so
            // this CPU<->GPU transfer is skipped on the ~N-1 subsequent
            // progressive-refinement calls.
            size_t aovSize = (size_t)options.width * (size_t)options.height * sizeof(float) * 4;
            if (aovs && (options.aovMask & kAovDepth) && aovs->depth)
            {
                ComPtr<ISlangBlob> rb;
                if (SLANG_SUCCEEDED(device->readBuffer(depthBuffer, 0, aovSize, rb.writeRef())) && rb)
                    memcpy(aovs->depth, rb->getBufferPointer(), aovSize);
            }
            if (aovs && (options.aovMask & kAovNormal) && aovs->normal)
            {
                ComPtr<ISlangBlob> rb;
                if (SLANG_SUCCEEDED(device->readBuffer(normalBuffer, 0, aovSize, rb.writeRef())) && rb)
                    memcpy(aovs->normal, rb->getBufferPointer(), aovSize);
            }
            if (aovs && (options.aovMask & kAovPrimId) && aovs->primId)
            {
                ComPtr<ISlangBlob> rb;
                if (SLANG_SUCCEEDED(device->readBuffer(primIdBuffer, 0, aovSize, rb.writeRef())) && rb)
                    memcpy(aovs->primId, rb->getBufferPointer(), aovSize);
            }
        }

        // M3 validation-only: PathTraceTestMain.slang dispatch. See VulkanRenderer.h.
        void RunPathTraceTest(const GpuPathTraceTestCase *cases, GpuPathTraceTestResult *results, int count)
        {
            UploadScene();

            struct CountParams { uint32_t count; } frame{(uint32_t)count};

            BufferDesc frameDesc = {};
            frameDesc.size = sizeof(frame);
            frameDesc.elementSize = sizeof(frame);
            frameDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            frameDesc.defaultState = ResourceState::ShaderResource;
            frameDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> frameBuf;
            if (SLANG_FAILED(device->createBuffer(frameDesc, &frame, frameBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create PathTrace test frame buffer\n");
                return;
            }

            BufferDesc casesDesc = {};
            casesDesc.size = (size_t)count * sizeof(GpuPathTraceTestCase);
            casesDesc.elementSize = sizeof(GpuPathTraceTestCase);
            casesDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            casesDesc.defaultState = ResourceState::ShaderResource;
            casesDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> casesBuf;
            if (SLANG_FAILED(device->createBuffer(casesDesc, cases, casesBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create PathTrace test cases buffer\n");
                return;
            }

            BufferDesc resultsDesc = {};
            resultsDesc.size = (size_t)count * sizeof(GpuPathTraceTestResult);
            resultsDesc.elementSize = sizeof(GpuPathTraceTestResult);
            resultsDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                                 BufferUsage::CopyDestination | BufferUsage::CopySource;
            resultsDesc.defaultState = ResourceState::UnorderedAccess;
            resultsDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> resultsBuf;
            if (SLANG_FAILED(device->createBuffer(resultsDesc, nullptr, resultsBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create PathTrace test results buffer\n");
                return;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(pathTraceTestPipeline);

                ShaderCursor cursor(rootObject);
                cursor["g_Frame"].setBinding(frameBuf);
                cursor["g_Cases"].setBinding(casesBuf);
                cursor["g_Results"].setBinding(resultsBuf);
                cursor["g_Primitives"].setBinding(primitivesBuf);
                cursor["g_MeshDescriptors"].setBinding(meshDescBuf);
                cursor["g_SceneBVH"].setBinding(sceneBvhBuf);
                cursor["g_MeshBVHNodes"].setBinding(meshBvhBuf);
                cursor["g_Positions"].setBinding(positionsBuf);
                cursor["g_PositionsEnd"].setBinding(positionsEndBuf);
                cursor["g_Tangents"].setBinding(tangentsBuf);
                cursor["g_Normals"].setBinding(normalsBuf);
                cursor["g_Indices"].setBinding(indicesBuf);
                cursor["g_UVs"].setBinding(uvsBuf);
                cursor["g_Materials"].setBinding(materialsBuf);
                BindMaterialTextures(cursor);
                cursor["g_TextureInfo"].setBinding(textureInfoBuf);
                cursor["g_Sky"].setBinding(skyBuf);
                cursor["g_Lights"].setBinding(lightsBuf);
                cursor["g_ProbeTexture"].setBinding(Binding(probeTexture, probeSampler));
                cursor["g_ProbeData"].setBinding(probeDataBuf);
                cursor["g_ProbeCdfX"].setBinding(probeCdfXBuf);
                cursor["g_ProbePdfX"].setBinding(probePdfXBuf);
                cursor["g_ProbeCdfY"].setBinding(probeCdfYBuf);
                cursor["g_ProbePdfY"].setBinding(probePdfYBuf);

                uint32_t groups = ((uint32_t)count + 63) / 64;
                passEncoder->dispatchCompute(groups, 1, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(resultsBuf, 0, resultsDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back PathTrace test results\n");
                return;
            }
            memcpy(results, readback->getBufferPointer(), resultsDesc.size);
        }

        // M4 validation-only: ProbeTestMain.slang dispatch. See VulkanRenderer.h.
        // Calls UploadScene() (via UploadProbe()) so the probe texture/CDF/PDF
        // buffers reflect *scene's sky.probe - no other scene buffers are
        // touched by ProbeTestMain, but uploading them anyway is harmless.
        void RunProbeTest(const GpuProbeTestCase *cases, GpuProbeTestResult *results, int count)
        {
            UploadScene();

            struct ProbeParams { int32_t width, height; uint32_t count; } frame{probeWidth, probeHeight, (uint32_t)count};

            BufferDesc frameDesc = {};
            frameDesc.size = sizeof(frame);
            frameDesc.elementSize = sizeof(frame);
            frameDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            frameDesc.defaultState = ResourceState::ShaderResource;
            frameDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> frameBuf;
            if (SLANG_FAILED(device->createBuffer(frameDesc, &frame, frameBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create Probe test frame buffer\n");
                return;
            }

            BufferDesc casesDesc = {};
            casesDesc.size = (size_t)count * sizeof(GpuProbeTestCase);
            casesDesc.elementSize = sizeof(GpuProbeTestCase);
            casesDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            casesDesc.defaultState = ResourceState::ShaderResource;
            casesDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> casesBuf;
            if (SLANG_FAILED(device->createBuffer(casesDesc, cases, casesBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create Probe test cases buffer\n");
                return;
            }

            BufferDesc resultsDesc = {};
            resultsDesc.size = (size_t)count * sizeof(GpuProbeTestResult);
            resultsDesc.elementSize = sizeof(GpuProbeTestResult);
            resultsDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                                 BufferUsage::CopyDestination | BufferUsage::CopySource;
            resultsDesc.defaultState = ResourceState::UnorderedAccess;
            resultsDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> resultsBuf;
            if (SLANG_FAILED(device->createBuffer(resultsDesc, nullptr, resultsBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create Probe test results buffer\n");
                return;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(probeTestPipeline);

                ShaderCursor cursor(rootObject);
                cursor["g_Frame"].setBinding(frameBuf);
                cursor["g_Cases"].setBinding(casesBuf);
                cursor["g_Results"].setBinding(resultsBuf);
                cursor["g_ProbeTexture"].setBinding(Binding(probeTexture, probeSampler));
                cursor["g_ProbeData"].setBinding(probeDataBuf);
                cursor["g_ProbeCdfX"].setBinding(probeCdfXBuf);
                cursor["g_ProbePdfX"].setBinding(probePdfXBuf);
                cursor["g_ProbeCdfY"].setBinding(probeCdfYBuf);
                cursor["g_ProbePdfY"].setBinding(probePdfYBuf);

                uint32_t groups = ((uint32_t)count + 63) / 64;
                passEncoder->dispatchCompute(groups, 1, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(resultsBuf, 0, resultsDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back Probe test results\n");
                return;
            }
            memcpy(results, readback->getBufferPointer(), resultsDesc.size);
        }

        // M0 validation kernel: proves device creation, module import
        // resolution, pipeline creation, dispatch and readback all work end to
        // end. Kept as a construction-time self-test; RunNormals() above is the
        // real M1 dispatch path.
        void RunClearColor(int w, int h, Color *output)
        {
            struct FrameParams
            {
                uint32_t width;
                uint32_t height;
            } frame{(uint32_t)w, (uint32_t)h};

            BufferDesc frameBufferDesc = {};
            frameBufferDesc.size = sizeof(frame);
            frameBufferDesc.elementSize = sizeof(frame);
            frameBufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            frameBufferDesc.defaultState = ResourceState::ShaderResource;
            frameBufferDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> frameBuffer;
            if (SLANG_FAILED(device->createBuffer(frameBufferDesc, &frame, frameBuffer.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create frame params buffer\n");
                return;
            }

            BufferDesc bufferDesc = {};
            bufferDesc.size = (size_t)w * (size_t)h * sizeof(float) * 4;
            bufferDesc.format = Format::Undefined;
            bufferDesc.elementSize = sizeof(float) * 4;
            bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                                BufferUsage::CopyDestination | BufferUsage::CopySource;
            bufferDesc.defaultState = ResourceState::UnorderedAccess;
            bufferDesc.memoryType = MemoryType::DeviceLocal;

            ComPtr<IBuffer> outputBuffer;
            if (SLANG_FAILED(device->createBuffer(bufferDesc, nullptr, outputBuffer.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create output buffer\n");
                return;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(clearColorPipeline);

                ShaderCursor(rootObject)["g_Frame"].setBinding(frameBuffer);
                ShaderCursor(rootObject)["g_Output"].setBinding(outputBuffer);

                uint32_t groupsX = (uint32_t(w) + 7) / 8;
                uint32_t groupsY = (uint32_t(h) + 7) / 8;
                passEncoder->dispatchCompute(groupsX, groupsY, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(outputBuffer, 0, bufferDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back output buffer\n");
                return;
            }

            memcpy(output, readback->getBufferPointer(), bufferDesc.size);
        }

        // M2 validation-only: BsdfTestMain.slang dispatch. See VulkanRenderer.h.
        void RunBsdfTest(const GpuBsdfTestCase *cases, GpuBsdfTestResult *results, int count)
        {
            struct CountParams { uint32_t count; } frame{(uint32_t)count};

            BufferDesc frameDesc = {};
            frameDesc.size = sizeof(frame);
            frameDesc.elementSize = sizeof(frame);
            frameDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            frameDesc.defaultState = ResourceState::ShaderResource;
            frameDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> frameBuf;
            if (SLANG_FAILED(device->createBuffer(frameDesc, &frame, frameBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create BSDF test frame buffer\n");
                return;
            }

            BufferDesc casesDesc = {};
            casesDesc.size = (size_t)count * sizeof(GpuBsdfTestCase);
            casesDesc.elementSize = sizeof(GpuBsdfTestCase);
            casesDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            casesDesc.defaultState = ResourceState::ShaderResource;
            casesDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> casesBuf;
            if (SLANG_FAILED(device->createBuffer(casesDesc, cases, casesBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create BSDF test cases buffer\n");
                return;
            }

            BufferDesc resultsDesc = {};
            resultsDesc.size = (size_t)count * sizeof(GpuBsdfTestResult);
            resultsDesc.elementSize = sizeof(GpuBsdfTestResult);
            resultsDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                                 BufferUsage::CopyDestination | BufferUsage::CopySource;
            resultsDesc.defaultState = ResourceState::UnorderedAccess;
            resultsDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> resultsBuf;
            if (SLANG_FAILED(device->createBuffer(resultsDesc, nullptr, resultsBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create BSDF test results buffer\n");
                return;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(bsdfTestPipeline);

                ShaderCursor cursor(rootObject);
                cursor["g_Frame"].setBinding(frameBuf);
                cursor["g_Cases"].setBinding(casesBuf);
                cursor["g_Results"].setBinding(resultsBuf);

                uint32_t groups = ((uint32_t)count + 63) / 64;
                passEncoder->dispatchCompute(groups, 1, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(resultsBuf, 0, resultsDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back BSDF test results\n");
                return;
            }
            memcpy(results, readback->getBufferPointer(), resultsDesc.size);
        }

        // M5 validation-only: TextureArraySpike.slang dispatch - the go/no-go
        // check for binding an *array* of combined image samplers (as opposed
        // to the single g_ProbeTexture binding this codebase already proves
        // out) via slang-rhi's ShaderCursor, ahead of building the real
        // per-material g_Textures[] array in M8. Creates kSpikeTextureCount
        // distinct 1x1 textures with known solid colors, binds all of them via
        // cursor["g_Textures"][i], dispatches, and returns the sampled colors
        // for the caller to compare against what was uploaded.
        static const uint32_t kSpikeTextureCount = VulkanRenderer::kTextureArraySpikeCount;

        bool RunTextureArraySpike(GpuTextureArraySpikeResult (&outColors)[kSpikeTextureCount])
        {
            static const float4_ kSpikeColors[kSpikeTextureCount] = {
                {1.0f, 0.0f, 0.0f, 1.0f},
                {0.0f, 1.0f, 0.0f, 1.0f},
                {0.0f, 0.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 0.0f, 1.0f},
            };

            ComPtr<ITexture> textures[kSpikeTextureCount];
            for (uint32_t i = 0; i < kSpikeTextureCount; ++i)
            {
                TextureDesc texDesc = {};
                texDesc.type = TextureType::Texture2D;
                texDesc.format = Format::RGBA32Float;
                texDesc.size = {1, 1, 1};
                texDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
                texDesc.memoryType = MemoryType::DeviceLocal;

                SubresourceData sub = {};
                sub.data = &kSpikeColors[i];
                sub.rowPitch = sizeof(float4_);
                sub.slicePitch = 0;
                if (SLANG_FAILED(device->createTexture(texDesc, &sub, textures[i].writeRef())))
                {
                    fprintf(stderr, "VulkanRenderer: M5 spike failed to create texture %u\n", i);
                    return false;
                }
            }

            ComPtr<ISampler> sampler;
            SamplerDesc samplerDesc = {};
            samplerDesc.minFilter = TextureFilteringMode::Linear;
            samplerDesc.magFilter = TextureFilteringMode::Linear;
            samplerDesc.addressU = TextureAddressingMode::Wrap;
            samplerDesc.addressV = TextureAddressingMode::Wrap;
            samplerDesc.addressW = TextureAddressingMode::Wrap;
            device->createSampler(samplerDesc, sampler.writeRef());

            BufferDesc outputDesc = {};
            outputDesc.size = kSpikeTextureCount * sizeof(float4_);
            outputDesc.elementSize = sizeof(float4_);
            outputDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                                BufferUsage::CopyDestination | BufferUsage::CopySource;
            outputDesc.defaultState = ResourceState::UnorderedAccess;
            outputDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> outputBuf;
            if (SLANG_FAILED(device->createBuffer(outputDesc, nullptr, outputBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: M5 spike failed to create output buffer\n");
                return false;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(textureArraySpikePipeline);

                ShaderCursor cursor(rootObject);
                ShaderCursor texturesCursor = cursor["g_Textures"];
                for (uint32_t i = 0; i < kSpikeTextureCount; ++i)
                    texturesCursor[i].setBinding(Binding(textures[i], sampler));
                cursor["g_Output"].setBinding(outputBuf);

                passEncoder->dispatchCompute(1, 1, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(outputBuf, 0, outputDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: M5 spike failed to read back output buffer\n");
                return false;
            }
            memcpy(outColors, readback->getBufferPointer(), outputDesc.size);
            return true;
        }

        // M8 validation-only: TextureSampleTestMain.slang dispatch - the GPU
        // side of kernel_validate's Material_SampleAlbedo/Roughness/Metallic
        // comparison against TextureSampling.h's CPU implementation. Builds
        // its own small g_Materials buffer from the caller-supplied
        // GpuMaterial array (whose texture indices reference *scene's
        // textures) via UploadMaterialTextures() directly - not the full
        // UploadScene(), since this test needs no primitives/BVH/camera.
        bool RunTextureSampleTest(const GpuMaterial *materials, int materialCount,
                                   const GpuTextureSampleTestCase *cases, GpuTextureSampleTestResult *results, int count)
        {
            UploadMaterialTextures();
            // Material.slang's Material_Sample* functions now reference
            // g_TextureInfo (UDIM remap) unconditionally - this standalone
            // test bypasses UploadScene() (where that buffer is normally
            // created), so it needs its own copy here.
            ComPtr<IBuffer> testTextureInfoBuf = CreateStructuredBuffer(textureInfos);

            std::vector<GpuMaterial> materialsVec(materials, materials + materialCount);
            ComPtr<IBuffer> testMaterialsBuf = CreateStructuredBuffer(materialsVec);

            struct CountParams { uint32_t count; } frame{(uint32_t)count};
            BufferDesc frameDesc = {};
            frameDesc.size = sizeof(frame);
            frameDesc.elementSize = sizeof(frame);
            frameDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            frameDesc.defaultState = ResourceState::ShaderResource;
            frameDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> frameBuf;
            if (SLANG_FAILED(device->createBuffer(frameDesc, &frame, frameBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create TextureSampleTest frame buffer\n");
                return false;
            }

            BufferDesc casesDesc = {};
            casesDesc.size = (size_t)count * sizeof(GpuTextureSampleTestCase);
            casesDesc.elementSize = sizeof(GpuTextureSampleTestCase);
            casesDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopyDestination;
            casesDesc.defaultState = ResourceState::ShaderResource;
            casesDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> casesBuf;
            if (SLANG_FAILED(device->createBuffer(casesDesc, cases, casesBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create TextureSampleTest cases buffer\n");
                return false;
            }

            BufferDesc resultsDesc = {};
            resultsDesc.size = (size_t)count * sizeof(GpuTextureSampleTestResult);
            resultsDesc.elementSize = sizeof(GpuTextureSampleTestResult);
            resultsDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess |
                                 BufferUsage::CopyDestination | BufferUsage::CopySource;
            resultsDesc.defaultState = ResourceState::UnorderedAccess;
            resultsDesc.memoryType = MemoryType::DeviceLocal;
            ComPtr<IBuffer> resultsBuf;
            if (SLANG_FAILED(device->createBuffer(resultsDesc, nullptr, resultsBuf.writeRef())))
            {
                fprintf(stderr, "VulkanRenderer: failed to create TextureSampleTest results buffer\n");
                return false;
            }

            {
                auto queue = device->getQueue(QueueType::Graphics);
                auto commandEncoder = queue->createCommandEncoder();
                auto passEncoder = commandEncoder->beginComputePass();
                auto rootObject = passEncoder->bindPipeline(textureSampleTestPipeline);

                ShaderCursor cursor(rootObject);
                cursor["g_Frame"].setBinding(frameBuf);
                cursor["g_Materials"].setBinding(testMaterialsBuf);
                cursor["g_Cases"].setBinding(casesBuf);
                cursor["g_Results"].setBinding(resultsBuf);
                BindMaterialTextures(cursor);
                cursor["g_TextureInfo"].setBinding(testTextureInfoBuf);

                uint32_t groups = ((uint32_t)count + 63) / 64;
                passEncoder->dispatchCompute(groups, 1, 1);
                passEncoder->end();

                queue->submit(commandEncoder->finish());
                queue->waitOnHost();
            }

            ComPtr<ISlangBlob> readback;
            if (SLANG_FAILED(device->readBuffer(resultsBuf, 0, resultsDesc.size, readback.writeRef())) || !readback)
            {
                fprintf(stderr, "VulkanRenderer: failed to read back TextureSampleTest results\n");
                return false;
            }
            memcpy(results, readback->getBufferPointer(), resultsDesc.size);
            return true;
        }
    };

    VulkanRenderer::VulkanRenderer(const Scene *s) : impl(new Impl())
    {
        impl->scene = s;
        impl->available = impl->CreateDevice() &&
                           impl->CreatePipeline("ClearColor", impl->clearColorPipeline) &&
                           impl->CreatePipeline("NormalsMain", impl->normalsPipeline) &&
                           impl->CreatePipeline("BsdfTestMain", impl->bsdfTestPipeline) &&
                           impl->CreatePipeline("PathTraceMain", impl->pathTracePipeline) &&
                           impl->CreatePipeline("PathTraceTestMain", impl->pathTraceTestPipeline) &&
                           impl->CreatePipeline("ProbeTestMain", impl->probeTestPipeline) &&
                           impl->CreatePipeline("TextureArraySpike", impl->textureArraySpikePipeline) &&
                           impl->CreatePipeline("TextureSampleTestMain", impl->textureSampleTestPipeline);
    }

    VulkanRenderer::~VulkanRenderer()
    {
        delete impl;
    }

    bool VulkanRenderer::IsAvailable() const
    {
        return impl->available;
    }

    void VulkanRenderer::Init(int width, int height)
    {
        impl->width = width;
        impl->height = height;
    }

    void VulkanRenderer::Render(const Camera &c, const Options &options, Color *output, AovBuffers *aovs)
    {
        if (!impl->available)
            return;

        if (options.mode == eNormals)
            impl->RunNormals(c, options, output);
        else if (options.mode == ePathTrace)
            impl->RunPathTrace(c, options, output, aovs);
        else
            impl->RunClearColor(impl->width, impl->height, output);
    }

    void VulkanRenderer::ResetAccumulation()
    {
        if (!impl->available)
            return;
        impl->ResetAccumulation();
    }

    void VulkanRenderer::RunBsdfTest(const GpuBsdfTestCase *cases, GpuBsdfTestResult *results, int count)
    {
        if (!impl->available)
            return;
        impl->RunBsdfTest(cases, results, count);
    }

    void VulkanRenderer::RunPathTraceTest(const GpuPathTraceTestCase *cases, GpuPathTraceTestResult *results, int count)
    {
        if (!impl->available)
            return;
        impl->RunPathTraceTest(cases, results, count);
    }

    void VulkanRenderer::RunProbeTest(const GpuProbeTestCase *cases, GpuProbeTestResult *results, int count)
    {
        if (!impl->available)
            return;
        impl->RunProbeTest(cases, results, count);
    }

    bool VulkanRenderer::RunTextureArraySpike(GpuTextureArraySpikeResult (&outColors)[kTextureArraySpikeCount])
    {
        if (!impl->available)
            return false;
        return impl->RunTextureArraySpike(outColors);
    }

    bool VulkanRenderer::RunTextureSampleTest(const GpuMaterial *materials, int materialCount,
                                               const GpuTextureSampleTestCase *cases, GpuTextureSampleTestResult *results, int count)
    {
        if (!impl->available)
            return false;
        return impl->RunTextureSampleTest(materials, materialCount, cases, results, count);
    }

    Renderer *CreateVulkanRenderer(const Scene *s)
    {
        return new VulkanRenderer(s);
    }
}
