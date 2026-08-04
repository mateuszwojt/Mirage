#pragma once

#include "mirage/math/Color.h"

namespace Mirage
{
    void NonLocalMeansFilter(const Color *in, Color *out, int width, int height, float falloff, int radius);
}