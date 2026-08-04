// Port of the per-ray-relevant subset of mirage/mirage/math/Bounds.h.
// Bounds transform (TransformBounds, needs Mat33) stays host-only: it is only used
// at BVH-build time (mirage/mirage/core/BVH.h), never from a per-ray GPU kernel.

struct Bounds
{
    float3 lower;
    float3 upper;

    __init(float3 lower_, float3 upper_)
    {
        lower = lower_;
        upper = upper_;
    }
};

float3 Bounds_GetCenter(Bounds b) { return 0.5f * (b.lower + b.upper); }
float3 Bounds_GetEdges(Bounds b) { return b.upper - b.lower; }

bool Bounds_Overlaps(Bounds b, float3 p)
{
    return !(p.x < b.lower.x || p.y < b.lower.y || p.z < b.lower.z ||
             p.x > b.upper.x || p.y > b.upper.y || p.z > b.upper.z);
}

bool Bounds_OverlapsBounds(Bounds a, Bounds b)
{
    return !(a.lower.x > b.upper.x || a.lower.y > b.upper.y || a.lower.z > b.upper.z ||
             a.upper.x < b.lower.x || a.upper.y < b.lower.y || a.upper.z < b.lower.z);
}

Bounds Bounds_Union(Bounds a, Bounds b)
{
    return Bounds(min(a.lower, b.lower), max(a.upper, b.upper));
}
