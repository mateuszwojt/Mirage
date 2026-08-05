// Mirrors mirage/shaders/NormalMapping.h's ApplyNormalMap - see that file
// for the full design rationale (why per-hit Gram-Schmidt, why not
// BasisFromVector, why the face-forward safety net checks against the
// smooth shading normal rather than the geometric triangle normal).

import MathExt;

float3 ApplyNormalMap(float3 normal, float3 tangent, float3 texel)
{
    // [0,1] -> [-1,1] decode. Standard (OpenGL-style) tangent-space
    // convention: texel (0.5,0.5,1.0) - the "flat", unperturbed resting
    // value - decodes to (0,0,1), i.e. straight along `normal`.
    float3 tangentSpaceNormal = texel * 2.0f - float3(1.0f, 1.0f, 1.0f);

    float3 t = SafeNormalize(tangent - normal * dot(normal, tangent), tangent);
    float3 b = cross(normal, t);

    float3 perturbed = SafeNormalize(
        t * tangentSpaceNormal.x + b * tangentSpaceNormal.y + normal * tangentSpaceNormal.z, normal);

    // Safety net against a normal map strong enough to flip the perturbed
    // normal onto the opposite side of the (already correctly
    // face-forwarded) unperturbed shading normal - see NormalMapping.h for
    // why this checks against the smooth shading normal, not the raw
    // geometric triangle normal.
    if (dot(perturbed, normal) <= 0.0f)
        return normal;

    return perturbed;
}
