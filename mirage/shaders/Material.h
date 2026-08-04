#pragma once

#include "mirage/shaders/Texture.h"
#include "mirage/math/Vec3.h"

#include <vector>

namespace Mirage
{
	class Material
	{
	public:
		Material()
		{
			color = Vec3(0.82f, 0.67f, 0.16f);
			emission = Vec3(0.0f);
			absorption = Vec3(0.0);

			// when eta is zero the index of refraction will be inferred from the specular component
			eta = 0.0f;

			metallic = 0.0f;
			subsurface = 0.0f;
			specular = 0.5f;
			roughness = 0.5f;
			specularTint = 0.0f;
			anisotropic = 0.0f;
			sheen = 0.0f;
			sheenTint = 0.0f;
			clearcoat = 0.0f;
			clearcoatGloss = 1.0f;
			transmission = 0.0f;
			bump = 0.0f;
			bumpTile = 10.0f;

			albedoTextureIndex = -1;
			roughnessTextureIndex = -1;
			metallicTextureIndex = -1;
		};

		CUDA_CALLABLE inline float GetIndexOfRefraction() const
		{
			if (eta == 0.0f)
				return 2.0f / (1.0f - sqrtf(0.08 * specular)) - 1.0f;
			else
				return eta;
		}

		Vec3 emission;
		Vec3 color;
		Vec3 absorption;

		float eta;
		float metallic;
		float subsurface;
		float specular;
		float roughness;
		float specularTint;
		float anisotropic;
		float sheen;
		float sheenTint;
		float clearcoat;
		float clearcoatGloss;
		float transmission;

		Texture albedoMap;

		Texture bumpMap;
		float bump;
		Vec3 bumpTile;

		// Indices into Scene::textures (-1 = none, use the scalar/color fields
		// above as-is). Populated by texture-loading paths (.tin scene format,
		// hdMirage's UsdPreviewSurface translation); sampled per-hit at shading
		// time to build a "shaded" Material copy - see TextureSampling.h.
		int albedoTextureIndex;
		int roughnessTextureIndex;
		int metallicTextureIndex;
	};
}