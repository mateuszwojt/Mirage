// Bit-exact port of Mirage::Random (mirage/mirage/utils/MathUtils.h).
// Kept algorithmically identical to the CPU RNG so per-thread streams are only
// distinguished by seeding (see PathTraceMain.slang), not by algorithm drift.

struct Random
{
    uint seed1;
    uint seed2;

    __init(uint seed)
    {
        seed1 = 315645664u + seed;
        seed2 = seed1 ^ 0x13ab45feu;
    }

    [mutating]
    uint Rand()
    {
        seed1 = (seed2 ^ ((seed1 << 5) | (seed1 >> 27))) ^ (seed1 * seed2);
        seed2 = seed1 ^ ((seed2 << 12) | (seed2 >> 20));
        return seed1;
    }

    [mutating]
    uint Rand(uint lo, uint hi)
    {
        return lo + Rand() % (hi - lo);
    }

    // Random float in [0, 1).
    [mutating]
    float Randf()
    {
        uint value = Rand();
        return float(value) * (1.0f / float(0xffffffffu));
    }

    // Random float in [lo, hi).
    [mutating]
    float Randf(float lo, float hi)
    {
        float t = Randf();
        return (1.0f - t) * lo + t * hi;
    }

    // Random float in [0, hi).
    [mutating]
    float Randf(float hi)
    {
        return Randf() * hi;
    }
};
