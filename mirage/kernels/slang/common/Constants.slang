// Shared numeric constants for Mirage kernels.
// Mirrors the scattered #define/const usages in mirage/mirage/utils/*.h and mirage/mirage/core/*.h.

static const float kPi = 3.14159265359f;
static const float kInvPi = 0.31830988618f;
static const float kTwoPi = 6.28318530718f;
static const float kInvTwoPi = 1.0f / 6.28318530718f;
static const float kRayEpsilon = 0.0001f;

// Fixed-size array bound for the per-material texture array (Material.slang's
// g_Textures) - not true bindless (see the plan's Deferrals: "True bindless
// descriptor indexing"). 32 is a generous cap relative to anything this
// codebase currently exercises, not a measured/derived constraint; exceeding
// it needs a recreate-with-larger-array follow-up, not built here.
static const uint kMaxMaterialTextures = 32;
