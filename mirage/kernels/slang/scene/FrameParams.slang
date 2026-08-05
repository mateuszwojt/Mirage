// Canonical per-dispatch uniform struct, byte-identical to GpuLayout.h's
// GpuFrameParams. Shared by every integrator entry point (NormalsMain.slang,
// PathTraceMain.slang) instead of each kernel declaring its own ad-hoc local
// mirror - a prior version of this codebase had NormalsMain.slang define its
// own smaller/stale local copy, which silently drifted out of sync with the
// real (larger) C++ struct once PathTraceMain's fields were added; this file
// is the fix.

struct GpuFrameParams
{
    uint width;
    uint height;
    uint numPrimitives;
    int maxDepth;
    uint frameSeed;
    uint sampleIndex;
    float clampValue;
    uint numLights; // Tier-2 "punctual lights" - appended, not interleaved
};
