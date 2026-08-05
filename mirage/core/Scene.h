#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <memory>

#include "mirage/prims/Primitive.h"
#include "mirage/prims/Mesh.h"
#include "mirage/prims/PointInstancer.h"
#include "mirage/camera/Camera.h"
#include "mirage/shaders/Texture.h"
#include "mirage/shaders/Material.h"
#include "mirage/utils/MathUtils.h"
#include "mirage/lights/PunctualLight.h"

#include "mirage/core/BVH.h"
#include "mirage/core/Probe.h"

namespace Mirage
{
	class Camera;

	struct Light : Primitive
	{
		Light() { type = eSphere; }

		Vec3 position;
		Vec3 emission;

		float area;
	};

	struct Sky : Light
	{
		Vec3 horizon;
		Vec3 zenith;

		Probe probe;

		CUDA_CALLABLE Vec3 Eval(const Vec3 &dir) const
		{
			if (probe.valid)
			{
				return ProbeEval(probe, ProbeDirToUV(dir));
			}
			else
			{
				return ::Mirage::Lerp(horizon, zenith, sqrtf(fabs(dir.y)));
			}
		}
	};

	class Scene
	{
	public:
		Scene() : camera(nullptr) {};
		~Scene() = default;

		// Camera
		std::unique_ptr<Camera> camera;

		// Primitives
		std::vector<Primitive> primitives;

		// Procedural point-instancers (Tier-1 "instancing/particles") -
		// expanded into ordinary `primitives` entries by Build(), see
		// ExpandInstancers(). Not itself an acceleration-structure input -
		// by the time Build() constructs the scene BVH, every instance is
		// already a plain Primitive like any other.
		std::vector<PointInstancer> instancers;

		// Point/directional punctual lights - deliberately not Primitives
		// (no area, no intersection geometry, never camera-visible or
		// BSDF-hittable) - see PunctualLight.h and SampleLights' dedicated
		// no-MIS NEE block for these.
		std::vector<PunctualLight> lights;

		// Meshes
		std::vector<std::unique_ptr<Mesh>> meshes;

		// Materials and textures, owned here and referenced by index -
		// Primitive::materialIndex indexes `materials`, Material::xxxTextureIndex
		// indexes `textures`. Not std::shared_ptr: nothing outside Scene needs
		// shared ownership, mirroring the existing `meshes` idiom (consumers that
		// need a stable pointer into a live scene, e.g. hdMirage's per-mesh
		// raw-pointer caching, take a raw pointer from GetMaterial()/GetTexture()
		// instead).
		std::vector<std::unique_ptr<Material>> materials;
		std::vector<std::unique_ptr<Texture>> textures;

		Sky sky;

		BVH bvh;

		void Clear();

		void AddPrimitive(const Primitive &prim);

		// Registers a PointInstancer - expanded into `primitives` the next
		// time Build() runs (see ExpandInstancers()). Order relative to
		// AddPrimitive() calls doesn't matter; expansion always happens
		// before the scene BVH is built.
		void AddInstancer(const PointInstancer &instancer);

		// Returns the new mesh's index in `meshes` (-1 if `mesh` was null,
		// same no-op-on-null behavior as before this return value existed) -
		// callers can stash it as MeshGeometry::meshIndex so
		// VulkanRenderer::UploadScene can resolve the primitive's
		// GpuMeshDescriptor in O(1) instead of an O(meshes) pointer scan.
		// Existing call sites that ignore the return value keep compiling
		// and behaving exactly as before.
		int AddMesh(std::unique_ptr<Mesh> mesh);

		// Always appends a new entry and returns its index - use when there is
		// no natural dedup key (e.g. kernel_validate's ad-hoc test scenes).
		int AddMaterial(std::unique_ptr<Material> mat);

		// Appends only if `name` hasn't been registered yet; returns the
		// existing index on a repeat name (e.g. the same .tin material block or
		// USD material path referenced by multiple primitives).
		int FindOrAddMaterial(const std::string &name, std::unique_ptr<Material> mat);

		// index < 0 or out of range returns nullptr - callers should treat an
		// unassigned/invalid materialIndex as "use Material()'s defaults", not
		// crash.
		Material *GetMaterial(int index) const;

		int AddTexture(std::unique_ptr<Texture> tex);

		// Appends only if `path` (the resolved, absolute file path) hasn't been
		// loaded yet; returns the existing index otherwise - this is what
		// prevents the same texture file referenced by two materials (or two
		// USD material re-syncs of the same file) from being decoded/uploaded
		// more than once.
		int FindOrAddTexture(const std::string &path, std::unique_ptr<Texture> tex);

		Texture *GetTexture(int index) const;

		void Build();

	private:
		std::mutex mutex;

		std::unordered_map<std::string, int> materialNameToIndex;
		std::unordered_map<std::string, int> texturePathToIndex;

		// Appends one Primitive per instance across every entry in
		// `instancers` into `primitives`, then clears `instancers` (so a
		// second Build() call - e.g. after AddPrimitive()'ing more geometry -
		// doesn't re-expand and duplicate the same instances). Called by
		// Build() while `mutex` is already held - not itself lock-acquiring,
		// to avoid re-entrant-locking the same non-recursive std::mutex.
		void ExpandInstancers();
	};
}