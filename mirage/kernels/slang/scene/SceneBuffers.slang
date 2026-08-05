// Global scene bindings - the Slang analogue of Renderer.cu's GPUScene, but
// generalized to "one mega-buffer per array type + per-primitive/mesh element
// offsets" instead of one CUDA texture object per mesh array (see the plan's
// "Data layout & marshalling" section).

import BVH;
import Quat;
import Material;

struct GpuTransform
{
    float3 p;
    Quat r;
    float s;
};

struct GpuPrimitive
{
    GpuTransform startTransform;
    GpuTransform endTransform;
    float4 unionData; // sphere: x=radius; plane: xyzw=plane eq
    int type;          // 0=sphere, 1=plane, 2=mesh
    int meshIndex;     // -1 if type != mesh
    int lightSamples;
    int materialIndex; // indexes g_Materials, one GpuMaterial per primitive
    int hydraId;        // stable id for the primId AOV, mirrors Primitive::hydraId
};

struct GpuMeshDescriptor
{
    uint vertexOffset, indexOffset, uvOffset, nodeOffset;
    uint numVertices, numIndices, numNodes;
    float area;
    uint hasUVs; // 0/1 - disambiguates "no UV buffer for this mesh" from "its UVs start at offset 0"
    // Deforming (2-keyframe) motion blur (Tier-1 feature 2): same
    // disambiguation pattern as hasUVs above - hasMotion == 0 means the mesh
    // is static and vertexOffsetEnd is meaningless. Mirrors Mesh::HasMotion()
    // on the CPU side.
    uint vertexOffsetEnd;
    uint hasMotion;
    // Tangent-space normal mapping (Tier-1 feature 5): same disambiguation
    // pattern again - hasTangents == 0 means the mesh has no tangent basis
    // and tangentOffset is meaningless. Mirrors Mesh::HasTangents().
    uint tangentOffset;
    uint hasTangents;
};

uniform StructuredBuffer<GpuPrimitive> g_Primitives;
uniform StructuredBuffer<GpuMeshDescriptor> g_MeshDescriptors;
uniform StructuredBuffer<BVHNode> g_SceneBVH;
uniform StructuredBuffer<BVHNode> g_MeshBVHNodes;
uniform StructuredBuffer<float3> g_Positions;
uniform StructuredBuffer<float3> g_PositionsEnd;
uniform StructuredBuffer<float3> g_Normals;
uniform StructuredBuffer<float3> g_Tangents;
uniform StructuredBuffer<int> g_Indices;
uniform StructuredBuffer<float2> g_UVs;
uniform StructuredBuffer<GpuMaterial> g_Materials;

// Progressive-accumulation buffer (rgb + weight), persists across dispatches -
// see PathTraceMain.slang and AddSample.slang. Not used by the M1/M2 kernels.
uniform RWStructuredBuffer<float4> g_AccumBuffer;

// First-hit AOV buffers - written unconditionally by PathTraceMain.slang every
// dispatch (cheap, a few extra stores per pixel); VulkanRenderer.cpp gates the
// actual host readback on Options::aovMask so the CPU<->GPU transfer only
// happens when an AOV is actually requested. Unlike g_AccumBuffer these are
// single-valued snapshots, not accumulated - see PathTrace.slang's outDepth/
// outNormal/outPrimId out-params.
uniform RWStructuredBuffer<float4> g_AovDepth;
uniform RWStructuredBuffer<float4> g_AovNormal;
uniform RWStructuredBuffer<float4> g_AovPrimId;
