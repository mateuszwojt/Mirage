// Port of the per-ray geometric kernels from mirage/mirage/core/Intersection.h
// actually needed by the M1 normals-only validation path (sphere/plane/tri/AABB
// tests). BVH traversal itself lives in Trace.slang since it needs to be
// threaded through the scene mega-buffers.

import MathExt;

bool SolveQuadratic(float a, float b, float c, out float minT, out float maxT)
{
    if (a == 0.0f && b == 0.0f)
    {
        minT = 0.0f;
        maxT = 0.0f;
        return true;
    }

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
    {
        minT = 0.0f;
        maxT = 0.0f;
        return false;
    }

    float sign = b < 0.0f ? -1.0f : 1.0f;
    float t = -0.5f * (b + sign * sqrt(discriminant));
    minT = t / a;
    maxT = c / t;

    if (maxT < minT)
    {
        float tmp = minT;
        minT = maxT;
        maxT = tmp;
    }

    return true;
}

// Mirrors IntersectRaySphere: returns closest hit t + world-space hit normal.
bool IntersectRaySphere(float3 sphereOrigin, float sphereRadius, float3 rayOrigin, float3 rayDir,
                         out float minT, out float maxT, out float3 hitNormal)
{
    float3 q = rayOrigin - sphereOrigin;

    float a = 1.0f;
    float b = 2.0f * dot(q, rayDir);
    float c = dot(q, q) - (sphereRadius * sphereRadius);

    hitNormal = float3(0.0f);
    bool r = SolveQuadratic(a, b, c, minT, maxT);

    if (minT < 0.0f && maxT < 0.0f)
        return false;

    if (minT < 0.0f && maxT > 0.0f)
        minT = maxT;

    if (r)
        hitNormal = normalize((rayOrigin + rayDir * minT) - sphereOrigin);

    return r;
}

// Mirrors IntersectRayPlane. plane is (a,b,c,d) with ax+by+cz+dw=0.
bool IntersectRayPlane(float3 p, float3 dir, float4 plane, out float t)
{
    float d = dot(plane, float4(dir, 0.0f));
    if (d == 0.0f)
    {
        t = 0.0f;
        return false;
    }
    t = -dot(plane, float4(p, 1.0f)) / d;
    return t > 0.0f;
}

// Mirrors IntersectRayAABBFast (the "Ompf" slab test used by BVH traversal).
bool IntersectRayAABBFast(float3 pos, float3 rcpDir, float3 lo, float3 hi, out float t)
{
    float l1 = (lo.x - pos.x) * rcpDir.x;
    float l2 = (hi.x - pos.x) * rcpDir.x;
    float lmin = min(l1, l2);
    float lmax = max(l1, l2);

    l1 = (lo.y - pos.y) * rcpDir.y;
    l2 = (hi.y - pos.y) * rcpDir.y;
    lmin = max(min(l1, l2), lmin);
    lmax = min(max(l1, l2), lmax);

    l1 = (lo.z - pos.z) * rcpDir.z;
    l2 = (hi.z - pos.z) * rcpDir.z;
    lmin = max(min(l1, l2), lmin);
    lmax = min(max(l1, l2), lmax);

    bool hit = (lmax >= 0.0f) && (lmax >= lmin);
    if (hit)
        t = lmin;
    else
        t = 0.0f;
    return hit;
}

// Mirrors IntersectRayTriTwoSided (Moller-Trumbore, two-sided).
bool IntersectRayTriTwoSided(float3 p, float3 dir, float3 a, float3 b, float3 c,
                              out float t, out float u, out float v, out float w,
                              out float sign, out float3 normal)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 n = cross(ab, ac);

    float d = dot(-dir, n);
    float ood = 1.0f / d;
    float3 ap = p - a;

    t = dot(ap, n) * ood;
    u = 0.0f; v = 0.0f; w = 0.0f; sign = d; normal = n;
    if (t < 0.0f)
        return false;

    float3 e = cross(-dir, ap);
    v = dot(ac, e) * ood;
    if (v < 0.0f || v > 1.0f)
        return false;
    w = -dot(ab, e) * ood;
    if (w < 0.0f || v + w > 1.0f)
        return false;

    u = 1.0f - v - w;
    return true;
}
