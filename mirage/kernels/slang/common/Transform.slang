// Port of mirage/mirage/math/Transform.h's per-ray-relevant subset.
// Used by the two-level BVH traversal to move rays into primitive-local space.

import Quat;

struct Transform
{
    float3 p;
    Quat r;
    float s;

    __init()
    {
        p = float3(0.0f);
        r = Quat();
        s = 1.0f;
    }

    __init(float3 p_, Quat r_, float s_)
    {
        p = p_; r = r_; s = s_;
    }
};

Transform Transform_Inverse(Transform t)
{
    Transform result;
    result.r = Quat_Conjugate(t.r);
    result.p = -Quat_Rotate(result.r, t.p);
    result.s = 1.0f / t.s;
    return result;
}

float3 TransformVector(Transform t, float3 v)
{
    return Quat_Rotate(t.r, t.s * v);
}

float3 TransformPoint(Transform t, float3 v)
{
    return t.p + Quat_Rotate(t.r, t.s * v);
}

float3 InverseTransformVector(Transform t, float3 v)
{
    return (1.0f / t.s) * Quat_Rotate(Quat_Conjugate(t.r), v);
}

float3 InverseTransformPoint(Transform t, float3 v)
{
    return (1.0f / t.s) * Quat_Rotate(Quat_Conjugate(t.r), v - t.p);
}

Transform InterpolateTransform(Transform a, Transform b, float t)
{
    float3 p = lerp(a.p, b.p, t);
    Quat r = Quat_Normalize(Quat_Lerp(a.r, b.r, t));
    float s = lerp(a.s, b.s, t);
    return Transform(p, r, s);
}
