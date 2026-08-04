// Port of prims/Primitive.h's Primitive dispatch + the mesh-local BVH traversal
// from core/Intersection.h's IntersectRayMesh/MeshQuery, generalized to index
// into the scene mega-buffers (SceneBuffers.slang) via GpuMeshDescriptor offsets
// instead of raw per-mesh pointers/CUDA texture objects.

import Quat;
import MathExt;
import Intersection;
import BVH;
import SceneBuffers;
import Sampler;
import Random;
import Constants;

float3 GpuTransform_TransformPoint(GpuTransform t, float3 v)
{
    return t.p + Quat_Rotate(t.r, t.s * v);
}

float3 GpuTransform_TransformVector(GpuTransform t, float3 v)
{
    return Quat_Rotate(t.r, t.s * v);
}

float3 GpuTransform_InverseTransformPoint(GpuTransform t, float3 v)
{
    return (1.0f / t.s) * Quat_Rotate(Quat_Conjugate(t.r), v - t.p);
}

float3 GpuTransform_InverseTransformVector(GpuTransform t, float3 v)
{
    return (1.0f / t.s) * Quat_Rotate(Quat_Conjugate(t.r), v);
}

GpuTransform GpuTransform_Interpolate(GpuTransform a, GpuTransform b, float t)
{
    GpuTransform result;
    result.p = lerp(a.p, b.p, t);
    result.r = Quat_Normalize(Quat_Lerp(a.r, b.r, t));
    result.s = lerp(a.s, b.s, t);
    return result;
}

// Mirrors IntersectRayMesh / MeshQuery: BVH traversal over the mesh-local node
// range (nodeOffset..nodeOffset+numNodes), closest-hit triangle test. rayTime
// drives the deforming-motion-blur vertex lerp below when desc.hasMotion != 0
// (mirrors MeshQuery::operator()'s mesh.positionsEnd branch on the CPU side).
bool IntersectRayMesh(GpuMeshDescriptor desc, float3 origin, float3 dir, float tmax, float rayTime,
                       out float outT, out float outU, out float outV, out float outW,
                       out int outTri, out float3 outNormal)
{
    float3 rcpDir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    int stack[32];
    stack[0] = 0;
    int count = 1;

    float closestT = tmax;
    float closestU = 0.0f, closestV = 0.0f, closestW = 0.0f;
    int closestTri = -1;
    float3 closestNormal = float3(0.0f);

    [loop]
    while (count > 0)
    {
        BVHNode node = FetchNode(g_MeshBVHNodes, desc.nodeOffset, stack[--count]);

        if (node.IsLeaf())
        {
            int tri = int(node.leftIndex);
            int i0 = g_Indices[desc.indexOffset + tri * 3 + 0];
            int i1 = g_Indices[desc.indexOffset + tri * 3 + 1];
            int i2 = g_Indices[desc.indexOffset + tri * 3 + 2];

            float3 a = g_Positions[desc.vertexOffset + i0];
            float3 b = g_Positions[desc.vertexOffset + i1];
            float3 c = g_Positions[desc.vertexOffset + i2];

            if (desc.hasMotion != 0)
            {
                a = lerp(a, g_PositionsEnd[desc.vertexOffsetEnd + i0], rayTime);
                b = lerp(b, g_PositionsEnd[desc.vertexOffsetEnd + i1], rayTime);
                c = lerp(c, g_PositionsEnd[desc.vertexOffsetEnd + i2], rayTime);
            }

            float t, u, v, w, sign;
            float3 n;
            if (IntersectRayTriTwoSided(origin, dir, a, b, c, t, u, v, w, sign, n))
            {
                if (t > 0.0f && t < closestT)
                {
                    closestT = t;
                    closestU = u; closestV = v; closestW = w;
                    closestTri = tri;
                    closestNormal = n * sign;
                }
            }
            tmax = closestT;
        }
        else
        {
            BVHNode left = FetchNode(g_MeshBVHNodes, desc.nodeOffset, node.leftIndex);
            BVHNode right = FetchNode(g_MeshBVHNodes, desc.nodeOffset, node.RightIndex());

            float tLeft, tRight;
            bool hitLeft = IntersectRayAABBFast(origin, rcpDir, left.lower, left.upper, tLeft) && tLeft < tmax;
            bool hitRight = IntersectRayAABBFast(origin, rcpDir, right.lower, right.upper, tRight) && tRight < tmax;

            uint leftIdx = node.leftIndex;
            uint rightIdx = node.RightIndex();
            if (hitLeft && hitRight && tLeft > tRight)
            {
                uint tmp = leftIdx; leftIdx = rightIdx; rightIdx = tmp;
                bool tmpB = hitLeft; hitLeft = hitRight; hitRight = tmpB;
            }

            if (hitLeft)
                stack[count++] = int(leftIdx);
            if (hitRight)
                stack[count++] = int(rightIdx);
        }
    }

    outT = closestT; outU = closestU; outV = closestV; outW = closestW;
    outTri = closestTri; outNormal = closestNormal;
    return closestTri >= 0;
}

// Mirrors PrimitiveIntersect's dispatch across sphere/plane/mesh. outUV is
// only populated for eMesh hits on a mesh with UV data (desc.hasUVs != 0) -
// left at float2(0.0f) otherwise, mirroring the CPU's PrimitiveIntersect
// outUV convention (Intersection.h) exactly. Sphere/plane hits never
// populate outUV - no procedural UV for non-mesh primitives in this
// milestone (see the plan's Deferrals).
bool PrimitiveIntersect(GpuPrimitive prim, float3 rayOrigin, float3 rayDir, float rayTime,
                         out float outT, out float3 outNormal, out float2 outUV)
{
    GpuTransform transform = GpuTransform_Interpolate(prim.startTransform, prim.endTransform, rayTime);
    outT = 0.0f;
    outNormal = float3(0.0f);
    outUV = float2(0.0f);

    if (prim.type == 0) // sphere
    {
        float minT, maxT;
        float3 n;
        bool hit = IntersectRaySphere(transform.p, prim.unionData.x * transform.s, rayOrigin, rayDir, minT, maxT, n);
        if (hit)
        {
            outT = minT;
            outNormal = n;
        }
        return hit;
    }
    else if (prim.type == 1) // plane
    {
        bool hit = IntersectRayPlane(rayOrigin, rayDir, prim.unionData, outT);
        if (hit)
            outNormal = prim.unionData.xyz;
        return hit;
    }
    else // mesh
    {
        if (prim.meshIndex < 0)
            return false;

        GpuMeshDescriptor desc = g_MeshDescriptors[prim.meshIndex];
        if (desc.numNodes == 0)
            return false;

        float3 localOrigin = GpuTransform_InverseTransformPoint(transform, rayOrigin);
        float3 localDir = GpuTransform_InverseTransformVector(transform, rayDir);

        float t, u, v, w;
        int tri;
        float3 triNormal;
        bool hit = IntersectRayMesh(desc, localOrigin, localDir, 3.402823466e+38f, rayTime, t, u, v, w, tri, triNormal);
        if (hit)
        {
            int i0 = g_Indices[desc.indexOffset + tri * 3 + 0];
            int i1 = g_Indices[desc.indexOffset + tri * 3 + 1];
            int i2 = g_Indices[desc.indexOffset + tri * 3 + 2];

            float3 n1 = g_Normals[desc.vertexOffset + i0];
            float3 n2 = g_Normals[desc.vertexOffset + i1];
            float3 n3 = g_Normals[desc.vertexOffset + i2];

            float3 smoothNormal = u * n1 + v * n2 + w * n3;
            if (dot(smoothNormal, triNormal) < 0.0f)
                smoothNormal *= -1.0f;

            outT = t;
            outNormal = SafeNormalize(GpuTransform_TransformVector(transform, smoothNormal), triNormal);

            // Same u/v/w barycentric weights (matched to i0/i1/i2) used for
            // the smooth-normal interpolation above, applied to per-vertex
            // UVs instead - mirrors Intersection.h's outUV block exactly.
            if (desc.hasUVs != 0)
            {
                float2 uv0 = g_UVs[desc.uvOffset + i0];
                float2 uv1 = g_UVs[desc.uvOffset + i1];
                float2 uv2 = g_UVs[desc.uvOffset + i2];
                outUV = u * uv0 + v * uv1 + w * uv2;
            }
        }
        return hit;
    }
}

// Mirrors PrimitiveArea's dispatch.
float PrimitiveArea(GpuPrimitive prim)
{
    if (prim.type == 0) // sphere
    {
        return 4.0f * kPi * prim.unionData.x * prim.unionData.x;
    }
    else if (prim.type == 1) // plane
    {
        return 0.0f;
    }
    else // mesh
    {
        if (prim.meshIndex < 0)
            return 0.0f;
        return g_MeshDescriptors[prim.meshIndex].area * prim.endTransform.s;
    }
}

// Mirrors PrimitiveSample. Mesh light sampling (CDF-based triangle picking) is
// deferred - it needs a per-mesh triangle-area CDF buffer that isn't uploaded
// yet (no test scene uses a mesh as a light source; see the plan's M3 notes).
void PrimitiveSample(GpuPrimitive prim, float time, inout Random rand, out float3 pos, out float3 normal)
{
    GpuTransform transform = GpuTransform_Interpolate(prim.startTransform, prim.endTransform, time);
    pos = float3(0.0f);
    normal = float3(0.0f, 1.0f, 0.0f);

    if (prim.type == 0) // sphere
    {
        float u1, u2;
        Sample2D(rand, u1, u2);
        pos = GpuTransform_TransformPoint(transform, UniformSampleSphere(u1, u2) * prim.unionData.x);
        normal = normalize(pos - transform.p);
    }
    // plane / mesh: not sampled as lights by any current test scene.
}
