#pragma once

#include <cstddef>
#include <vector>

namespace Mirage
{
	// `data` holds width*height*depth RGBA-interleaved texels (4 floats each,
	// linear color space regardless of the source file's encoding - see
	// TextureLoader.h). `depth` is always 1 for the 2D image textures
	// TextureLoader.h produces (albedo/roughness/metallic maps); it exists for
	// a possible future 3D-lattice use (e.g. a volumetric bump map) that was
	// never actually implemented in this codebase (see the plan's Deferrals -
	// bump mapping stays dead code) and has no current producer.
	class Texture
	{
	public:
		Texture() : data(nullptr), width(0), height(0), depth(0) {};
		~Texture() { delete[] data; }

		// Owns `data` (heap-allocated via new[]). Real (data-populated)
		// instances live in Scene::textures behind unique_ptr and are never
		// copied - but Material embeds a Texture by value (the still-dead
		// albedoMap/bumpMap fields, see the plan's Deferrals), and Material
		// itself is copied throughout the codebase, so Texture must stay a
		// well-behaved value type (deep copy, not shallow) rather than
		// disabling copying outright.
		Texture(const Texture &other)
			: data(other.data ? CopyData(other.data, other.width, other.height, other.depth) : nullptr),
			  width(other.width), height(other.height), depth(other.depth)
		{
		}

		Texture &operator=(const Texture &other)
		{
			if (this != &other)
			{
				float *newData = other.data ? CopyData(other.data, other.width, other.height, other.depth) : nullptr;
				delete[] data;
				data = newData;
				width = other.width;
				height = other.height;
				depth = other.depth;
			}
			return *this;
		}

		Texture(Texture &&other) noexcept
			: data(other.data), width(other.width), height(other.height), depth(other.depth)
		{
			other.data = nullptr;
			other.width = other.height = other.depth = 0;
		}

		Texture &operator=(Texture &&other) noexcept
		{
			if (this != &other)
			{
				delete[] data;
				data = other.data;
				width = other.width;
				height = other.height;
				depth = other.depth;
				other.data = nullptr;
				other.width = other.height = other.depth = 0;
			}
			return *this;
		}

		// Texture Data
		float* data;

		int width, height, depth;

	private:
		static float *CopyData(const float *src, int width, int height, int depth)
		{
			size_t count = (size_t)width * (size_t)height * (size_t)depth * 4;
			float *dst = new float[count];
			for (size_t i = 0; i < count; ++i)
				dst[i] = src[i];
			return dst;
		}
	};
}
