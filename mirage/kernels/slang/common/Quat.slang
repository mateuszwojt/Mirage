// Port of the Quat parts of mirage/mirage/math/Mat33.h.
// The Mat33-conversion constructor and Mat33 itself are NOT ported: on the host,
// Mat33/Quat<->Mat33 conversion is only used by BVH-build-time Bounds transforms
// (mirage/mirage/math/Bounds.h's TransformBounds), which stays a host preprocessing
// step and is never called from a per-ray GPU kernel.

struct Quat
{
    float x, y, z, w; // w is the real part

    __init()
    {
        x = 0.0f; y = 0.0f; z = 0.0f; w = 1.0f;
    }

    __init(float x_, float y_, float z_, float w_)
    {
        x = x_; y = y_; z = z_; w = w_;
    }
};

Quat Quat_Neg(Quat a) { return Quat(-a.x, -a.y, -a.z, -a.w); }
Quat Quat_Add(Quat a, Quat b) { return Quat(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
Quat Quat_Sub(Quat a, Quat b) { return Quat(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
Quat Quat_Scale(Quat a, float s) { return Quat(a.x * s, a.y * s, a.z * s, a.w * s); }

Quat Quat_Mul(Quat a, Quat b)
{
    return Quat(
        a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
        a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
        a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

Quat Quat_Normalize(Quat q)
{
    float length = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float rcpLength = 1.0f / length;
    return Quat_Scale(q, rcpLength);
}

Quat Quat_Conjugate(Quat q) { return Quat(-q.x, -q.y, -q.z, q.w); }

float3 Quat_Rotate(Quat q, float3 v)
{
    // q * v * q'
    Quat t = Quat_Mul(Quat_Mul(q, Quat(v.x, v.y, v.z, 0.0f)), Quat_Conjugate(q));
    return float3(t.x, t.y, t.z);
}

Quat Quat_Lerp(Quat a, Quat b, float t)
{
    return Quat(
        (1.0f - t) * a.x + t * b.x,
        (1.0f - t) * a.y + t * b.y,
        (1.0f - t) * a.z + t * b.z,
        (1.0f - t) * a.w + t * b.w);
}
