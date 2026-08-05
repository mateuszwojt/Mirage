// Port of Renderer.cpp's Trace(): top-level scene BVH traversal (QueryBVH),
// testing each leaf primitive via PrimitiveIntersect and keeping the closest hit.

import Constants;
import MathExt;
import BVH;
import Intersection;
import SceneBuffers;
import Primitive;
import Material;
import Random;

// outUV mirrors outNormal: populated only for eMesh hits on a UV-bearing
// mesh, left at float2(0.0f) otherwise (see PrimitiveIntersect's outUV
// comment). Unlike the CPU's Trace() (Renderer.cpp), this out parameter is
// not optional/defaultable - Slang has no equivalent of a nullable out
// pointer - so every call site (including SampleLights.slang's two shadow-ray
// calls, which don't need the value) must pass a local to receive it.
bool Trace(float3 rayOrigin, float3 rayDir, float rayTime, uint numScenePrims,
           out float outT, out float3 outNormal, out int outPrimIndex, out float2 outUV)
{
    float3 rcpDir = float3(1.0f / rayDir.x, 1.0f / rayDir.y, 1.0f / rayDir.z);

    int stack[32];
    stack[0] = 0;
    int count = 1;

    float minT = 3.402823466e+38f;
    float3 closestNormal = float3(0.0f);
    float2 closestUV = float2(0.0f);
    int closestPrim = -1;

    // Single-primitive scenes still get a valid (degenerate) BVH from the host
    // builder, but guard here too in case numScenePrims == 0 is ever reached.
    if (numScenePrims == 0)
    {
        outT = 0.0f; outNormal = float3(0.0f); outPrimIndex = -1; outUV = float2(0.0f);
        return false;
    }

    [loop]
    while (count > 0)
    {
        BVHNode node = FetchNode(g_SceneBVH, 0, stack[--count]);

        if (node.IsLeaf())
        {
            int primIndex = int(node.leftIndex);
            GpuPrimitive prim = g_Primitives[primIndex];

            float t;
            float3 n;
            float2 uv;
            if (PrimitiveIntersect(prim, rayOrigin, rayDir, rayTime, t, n, uv))
            {
                if (t < minT && t > 0.0f)
                {
                    minT = t;
                    closestNormal = n;
                    closestUV = uv;
                    closestPrim = primIndex;
                }
            }
        }
        else
        {
            BVHNode left = FetchNode(g_SceneBVH, 0, node.leftIndex);
            BVHNode right = FetchNode(g_SceneBVH, 0, node.RightIndex());

            float tLeft, tRight;
            bool hitLeft = IntersectRayAABBFast(rayOrigin, rcpDir, left.lower, left.upper, tLeft);
            bool hitRight = IntersectRayAABBFast(rayOrigin, rcpDir, right.lower, right.upper, tRight);

            if (hitLeft)
                stack[count++] = int(node.leftIndex);
            if (hitRight)
                stack[count++] = int(node.RightIndex());
        }
    }

    outT = minT;
    outNormal = FaceForward(closestNormal, -rayDir);
    outPrimIndex = closestPrim;
    outUV = closestUV;
    return closestPrim >= 0;
}

// Mirrors Renderer.cpp's TraceWithCutout(): traces a ray, transparently
// skipping past any surface that fails a stochastic opacity test
// (GpuMaterial.opacity/opacityTextureIndex) - "cutout" transparency
// (accept/reject against a uniform draw, not alpha blending). On a reject,
// the ray continues from just past the rejected hit point, without
// consuming caller-visible bounce/depth budget. Same signature as Trace()
// plus `rand`, so every call site swaps in directly. Bounded by
// kMaxCutoutRetries; if that budget runs out, the last-tested hit is
// reported as-is (treated as opaque) rather than retried forever.
static const int kMaxCutoutRetries = 4;

bool TraceWithCutout(float3 rayOrigin, float3 rayDir, float rayTime, uint numScenePrims, inout Random rand,
                      out float outT, out float3 outNormal, out int outPrimIndex, out float2 outUV)
{
    // outT must end up measured from the ORIGINAL ray origin the caller
    // passed in, not from whatever retry-shifted origin the final accepted
    // Trace() call used - SampleLights.slang compares outT against an
    // independently-computed distance-to-light from that same original
    // origin, so without this running total a retried ray would report a
    // too-short distance and wrongly treat a fully-visible light as
    // occluded by whatever it cutout-passed-through en route.
    float accumulatedT = 0.0f;
    outT = 0.0f;

    [loop]
    for (int i = 0; i < kMaxCutoutRetries; ++i)
    {
        float localT;
        if (!Trace(rayOrigin, rayDir, rayTime, numScenePrims, localT, outNormal, outPrimIndex, outUV))
            return false;

        GpuMaterial hitMaterial = g_Materials[g_Primitives[outPrimIndex].materialIndex];
        float opacity = hitMaterial.opacity;
        if (hitMaterial.opacityTextureIndex >= 0)
            opacity = Material_SampleOpacity(hitMaterial, outUV);

        outT = accumulatedT + localT;

        if (opacity >= 1.0f || rand.Randf() < opacity)
            return true;

        float3 p = rayOrigin + rayDir * localT;
        accumulatedT = outT + kRayEpsilon;
        rayOrigin = p + rayDir * kRayEpsilon;
    }

    return true; // retry budget exhausted - report the last hit as-is
}
