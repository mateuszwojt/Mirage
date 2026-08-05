#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>

namespace Mirage
{
	// `data` holds width*height*depth RGBA-interleaved texels (4 floats each,
	// linear color space regardless of the source file's encoding - see
	// TextureLoader.h). `depth` is always 1 for the 2D image textures
	// TextureLoader.h produces (albedo/roughness/metallic maps); it exists for
	// a possible future 3D-lattice use (e.g. a volumetric bump map) that was
	// never actually implemented in this codebase (see the plan's Deferrals -
	// bump mapping stays dead code) and has no current producer.
	//
	// Mip chain (Tier-2 "mipmapping/texture streaming"): mipData/mipWidths/
	// mipHeights hold a full, independent chain from level 0 (same content as
	// `data`/`width`/`height`, but a separate allocation - see GenerateMips())
	// down to 1x1, when populated. `data`/`width`/`height` are left completely
	// untouched by this feature - mipCount defaults to 1 and mipData stays
	// null until GenerateMips() runs, so every pre-existing call site that
	// never asks for a mip level keeps compiling and behaving unchanged.
	class Texture
	{
	public:
		Texture() : data(nullptr), width(0), height(0), depth(0) {};
		~Texture() { delete[] data; FreeMips(); }

		// Owns `data` (heap-allocated via new[]). Real (data-populated)
		// instances live in Scene::textures behind unique_ptr and are never
		// copied - but Material embeds a Texture by value (the still-dead
		// albedoMap/bumpMap fields, see the plan's Deferrals), and Material
		// itself is copied throughout the codebase, so Texture must stay a
		// well-behaved value type (deep copy, not shallow) rather than
		// disabling copying outright.
		Texture(const Texture &other)
			: data(other.data ? CopyData(other.data, other.width, other.height, other.depth) : nullptr),
			  width(other.width), height(other.height), depth(other.depth),
			  mipCount(0), mipData(nullptr), mipWidths(nullptr), mipHeights(nullptr),
			  udimGridWidth(other.udimGridWidth), udimGridHeight(other.udimGridHeight)
		{
			CopyMipsFrom(other);
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
				udimGridWidth = other.udimGridWidth;
				udimGridHeight = other.udimGridHeight;

				FreeMips();
				CopyMipsFrom(other);
			}
			return *this;
		}

		Texture(Texture &&other) noexcept
			: data(other.data), width(other.width), height(other.height), depth(other.depth),
			  mipCount(other.mipCount), mipData(other.mipData), mipWidths(other.mipWidths), mipHeights(other.mipHeights),
			  udimGridWidth(other.udimGridWidth), udimGridHeight(other.udimGridHeight)
		{
			other.data = nullptr;
			other.width = other.height = other.depth = 0;
			other.mipCount = 0;
			other.mipData = nullptr;
			other.mipWidths = nullptr;
			other.mipHeights = nullptr;
			other.udimGridWidth = other.udimGridHeight = 1;
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
				udimGridWidth = other.udimGridWidth;
				udimGridHeight = other.udimGridHeight;

				FreeMips();
				mipCount = other.mipCount;
				mipData = other.mipData;
				mipWidths = other.mipWidths;
				mipHeights = other.mipHeights;

				other.data = nullptr;
				other.width = other.height = other.depth = 0;
				other.mipCount = 0;
				other.mipData = nullptr;
				other.mipWidths = nullptr;
				other.mipHeights = nullptr;
				other.udimGridWidth = other.udimGridHeight = 1;
			}
			return *this;
		}

		// Texture Data
		float* data;

		int width, height, depth;

		// Mip chain - see the class comment above. mipCount == 0 means "never
		// generated" (distinct from mipCount == 1, "generated, but the base
		// level was already 1x1 so there's nothing to downsample") - sampling
		// code should treat both as "no mip data available, use level 0 (data/
		// width/height) directly".
		int mipCount = 0;
		float **mipData = nullptr;
		int *mipWidths = nullptr;
		int *mipHeights = nullptr;

		// UDIM atlas metadata (Tier-2 "UDIM"): both 1 (the default) means
		// "an ordinary single-tile texture, sample uv directly" - > 1 means
		// this texture is a composited atlas of udimGridWidth x
		// udimGridHeight UDIM tiles, and sampling must first remap uv into
		// atlas space (see UdimAtlasUV() in TextureSampling.h).
		int udimGridWidth = 1;
		int udimGridHeight = 1;

		// Box-filter-downsamples `data`/`width`/`height` into a full mip
		// chain (mipData[0] a separate copy of level 0, down to 1x1),
		// replacing any previously-generated chain. No-op (leaves mipCount
		// at 0) if `data` is null - matches the class's existing "null data
		// is a valid, harmless empty-texture state" convention (e.g. a
		// default-constructed Texture, or one whose file load failed).
		void GenerateMips()
		{
			FreeMips();

			if (!data || width <= 0 || height <= 0)
				return;

			int levels = 1;
			for (int w = width, h = height; w > 1 || h > 1; )
			{
				w = std::max(1, w / 2);
				h = std::max(1, h / 2);
				++levels;
			}

			mipData = new float *[levels];
			mipWidths = new int[levels];
			mipHeights = new int[levels];
			mipCount = levels;

			mipWidths[0] = width;
			mipHeights[0] = height;
			mipData[0] = CopyData(data, width, height, 1);

			for (int level = 1; level < levels; ++level)
			{
				int prevW = mipWidths[level - 1];
				int prevH = mipHeights[level - 1];
				int w = std::max(1, prevW / 2);
				int h = std::max(1, prevH / 2);

				const float *prev = mipData[level - 1];
				float *cur = new float[(size_t)w * (size_t)h * 4];

				for (int y = 0; y < h; ++y)
				{
					for (int x = 0; x < w; ++x)
					{
						// Box filter: average the (up to) 2x2 texels in the
						// previous level this texel downsamples from,
						// clamping at the edge for odd prevW/prevH (matches
						// the wrap-at-edge-via-clamp convention every other
						// sampler in this codebase uses for out-of-range
						// texel fetches).
						int x0 = std::min(x * 2, prevW - 1);
						int x1 = std::min(x * 2 + 1, prevW - 1);
						int y0 = std::min(y * 2, prevH - 1);
						int y1 = std::min(y * 2 + 1, prevH - 1);

						for (int c = 0; c < 4; ++c)
						{
							float sum = prev[(y0 * prevW + x0) * 4 + c] +
										prev[(y0 * prevW + x1) * 4 + c] +
										prev[(y1 * prevW + x0) * 4 + c] +
										prev[(y1 * prevW + x1) * 4 + c];
							cur[(y * w + x) * 4 + c] = sum * 0.25f;
						}
					}
				}

				mipWidths[level] = w;
				mipHeights[level] = h;
				mipData[level] = cur;
			}
		}

	private:
		static float *CopyData(const float *src, int width, int height, int depth)
		{
			size_t count = (size_t)width * (size_t)height * (size_t)depth * 4;
			float *dst = new float[count];
			for (size_t i = 0; i < count; ++i)
				dst[i] = src[i];
			return dst;
		}

		void FreeMips()
		{
			if (mipData)
			{
				for (int i = 0; i < mipCount; ++i)
					delete[] mipData[i];
				delete[] mipData;
			}
			delete[] mipWidths;
			delete[] mipHeights;
			mipData = nullptr;
			mipWidths = nullptr;
			mipHeights = nullptr;
			mipCount = 0;
		}

		void CopyMipsFrom(const Texture &other)
		{
			if (other.mipCount <= 0 || !other.mipData)
				return;

			mipCount = other.mipCount;
			mipData = new float *[mipCount];
			mipWidths = new int[mipCount];
			mipHeights = new int[mipCount];
			for (int i = 0; i < mipCount; ++i)
			{
				mipWidths[i] = other.mipWidths[i];
				mipHeights[i] = other.mipHeights[i];
				mipData[i] = CopyData(other.mipData[i], other.mipWidths[i], other.mipHeights[i], 1);
			}
		}
	};
}
