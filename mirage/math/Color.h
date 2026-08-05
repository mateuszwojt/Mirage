#pragma once

#include "mirage/math/Vec4.h"

namespace Mirage
{
	typedef Vec4 Color;

	CUDA_CALLABLE inline Color YxyToXYZ(float Y, float x, float y)
	{
		float X = x * (Y / y);
		float Z = (1.0f - x - y) * Y / y;

		return Color(X, Y, Z, 1.0f);
	}

	CUDA_CALLABLE inline Color HSVToRGB(float h, float s, float v)
	{
		float r, g, b;

		int i;
		float f, p, q, t;
		if (s == 0)
		{
			// achromatic (grey)
			r = g = b = v;
		}
		else
		{
			h *= 6.0f; // sector 0 to 5
			i = int(floor(h));
			f = h - i; // factorial part of h
			p = v * (1 - s);
			q = v * (1 - s * f);
			t = v * (1 - s * (1 - f));
			switch (i)
			{
			case 0:
				r = v;
				g = t;
				b = p;
				break;
			case 1:
				r = q;
				g = v;
				b = p;
				break;
			case 2:
				r = p;
				g = v;
				b = t;
				break;
			case 3:
				r = p;
				g = q;
				b = v;
				break;
			case 4:
				r = t;
				g = p;
				b = v;
				break;
			default: // case 5:
				r = v;
				g = p;
				b = q;
				break;
			};
		}

		return Color(r, g, b);
	}

	CUDA_CALLABLE inline Color XYZToLinear(float x, float y, float z)
	{
		float c[4];
		c[0] = 3.240479f * x + -1.537150f * y + -0.498535f * z;
		c[1] = -0.969256f * x + 1.875991f * y + 0.041556f * z;
		c[2] = 0.055648f * x + -0.204043f * y + 1.057311f * z;
		c[3] = 1.0f;

		return Color(c[0], c[1], c[2], c[3]);
	}

	CUDA_CALLABLE inline unsigned int ColorToRGBA8(const Color &c)
	{
		union SmallColor
		{
			uint8_t u8[4];
			int u32;
		};

		SmallColor s;
		s.u8[0] = (uint8_t)(::Mirage::Clamp(c.x, 0.0f, 1.0f) * 255);
		s.u8[1] = (uint8_t)(::Mirage::Clamp(c.y, 0.0f, 1.0f) * 255);
		s.u8[2] = (uint8_t)(::Mirage::Clamp(c.z, 0.0f, 1.0f) * 255);
		s.u8[3] = (uint8_t)(::Mirage::Clamp(c.w, 0.0f, 1.0f) * 255);

		return s.u32;
	}

	// Correct piecewise sRGB OETF (encode: linear -> display) / EOTF (decode:
	// display -> linear) - the IEC 61966-2-1 curve, not a naive pow(c, 2.2)/
	// pow(c, 1/2.2) gamma approximation. This used to be duplicated as two
	// disagreeing implementations - a correct piecewise decode private to
	// mirage/shaders/TextureLoader.cpp, and a separate naive-pow pair here -
	// unified here as the one sRGB implementation both now share (see
	// docs/PRODUCTION_READINESS.md's Tier 3 "no color management" finding).
	CUDA_CALLABLE inline float LinearToSrgb(float c)
	{
		if (c <= 0.0031308f)
			return c * 12.92f;
		return 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
	}

	CUDA_CALLABLE inline float SrgbToLinear(float c)
	{
		if (c <= 0.04045f)
			return c / 12.92f;
		return powf((c + 0.055f) / 1.055f, 2.4f);
	}

	CUDA_CALLABLE inline Color LinearToSrgb(const Color &c)
	{
		return Color(LinearToSrgb(c.x), LinearToSrgb(c.y), LinearToSrgb(c.z), c.w);
	}

	CUDA_CALLABLE inline Color SrgbToLinear(const Color &c)
	{
		return Color(SrgbToLinear(c.x), SrgbToLinear(c.y), SrgbToLinear(c.z), c.w);
	}

	CUDA_CALLABLE inline float Luminance(const Color &c)
	{
		return c.x * 0.3f + c.y * 0.6f + c.z * 0.1f;
	}

} // namespace Mirage
