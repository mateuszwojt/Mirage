// Non-builtin math helpers, mirroring mirage/mirage/math/Vec3.h and
// mirage/mirage/math/Geometric.h. Everything Slang already provides natively
// (dot, cross, normalize, mul, lerp, min, max, ...) is used directly instead of
// being re-wrapped here.

float3 SafeNormalize(float3 a, float3 fallback)
{
    float m = dot(a, a);
    if (m > 0.0f)
        return a * (1.0f / sqrt(m));
    return fallback;
}

float3 SafeNormalize(float3 a)
{
    return SafeNormalize(a, float3(0.0f));
}

float3 FaceForward(float3 n, float3 v)
{
    if (dot(v, n) < 0.0f)
        return -n;
    return n;
}

// Port of utils/MathUtils.h's ClampLength.
float3 ClampLength(float3 v, float maxLength)
{
    float l = length(v);
    if (l > maxLength)
        return v * (maxLength / l);
    return v;
}

// Port of math/Geometric.h's BasisFromVector (PBRT-style tangent frame from a
// normal). Used to build the (U,V,N) frame BSDFSample samples directions in.
void BasisFromVector(float3 w, out float3 u, out float3 v)
{
    if (abs(w.x) > abs(w.y))
    {
        float invLen = 1.0f / sqrt(w.x * w.x + w.z * w.z);
        u = float3(-w.z * invLen, 0.0f, w.x * invLen);
    }
    else
    {
        float invLen = 1.0f / sqrt(w.y * w.y + w.z * w.z);
        u = float3(0.0f, w.z * invLen, -w.y * invLen);
    }
    v = cross(w, u);
}
