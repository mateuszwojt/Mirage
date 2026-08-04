# Mirage Path Tracer

Mirage is a physically based path tracer, originally based on Miles Macklin's
Tinsel renderer (https://github.com/mmacklin/tinsel), shading with an
adaptation of Disney's principled BSDF.

Mirage is built as a static library. There's no standalone GUI application;
it's consumed either through the small CLI tools in `tools/`, or embedded in
a host application (e.g. a USD Hydra render delegate — see "Hydra
compatibility" below).

## Renderer backends

- **CPU** (`CreateCpuRenderer`) — reference path tracer, optionally
  multithreaded with OpenMP.
- **GPU, Vulkan/MoltenVK** (`CreateVulkanRenderer`) — the same integrator
  ported to [Slang](https://github.com/shader-slang/slang), compiled to
  SPIR-V at build time and run through
  [slang-rhi](https://github.com/shader-slang/slang-rhi)'s Vulkan backend
  (MoltenVK on macOS). This is currently the only GPU path — **CUDA support
  has been removed**, and the Vulkan/Slang kernel path is currently only
  wired up for macOS on Apple Silicon (arm64).
- `CreateNullRenderer` — no-op renderer, useful as a placeholder/for testing.

Both the CPU and Vulkan backends support three render modes (`RenderMode`:
`eNormals`, `eComplexity`, `ePathTrace`), optional depth-of-field and camera
motion blur (per-primitive start/end transforms plus shutter open/close),
and auxiliary AOV output (depth, world-space normal, and a stable prim ID)
alongside the beauty pass.

`CreateCpuWavefrontRenderer` and `CreateGpuWavefrontRenderer` are declared in
the public API but not yet implemented.

## Scene features

- Primitives: sphere, plane, and triangle mesh, accelerated with a BVH.
- Materials: Disney principled BSDF (metallic, roughness, specular,
  clearcoat, sheen, anisotropy, subsurface, transmission, etc.), with
  optional albedo/roughness/metallic texture maps (sRGB or linear).
- Lighting: explicit emissive primitives (next-event estimation), plus a sky
  that's either a simple horizon/zenith gradient or an HDR environment probe
  with importance sampling and MIS.
- A non-local-means denoising filter (`mirage/filter/NLM.h`) for the CPU
  path.

## Hydra compatibility

`mirage/core/HydraCompatibility.h` provides scaffolding
(`HydraCompatibleRenderer`, a resource registry, render-task tokens) for
wrapping Mirage as a USD Hydra render delegate backend. It currently wraps
`CreateCpuRenderer` internally. The `Material`/`Primitive` structs also carry
Hydra-specific fields (e.g. `Primitive::hydraId`) so a delegate can correlate
Mirage's primId AOV with the host's own picking/selection IDs. The delegate
itself (e.g. an `hdMirage` plugin) lives outside this repository and links
against the installed `Mirage::Mirage` CMake target.

## Tools

- **`scene_renderer`** — CLI that parses a small text scene format
  (materials + primitives, with optional texture maps), builds a `Scene`,
  path-traces it on the CPU, and writes the result to PNG/JPG/BMP/TGA:
  ```
  scene_renderer <scene_file> <output_image> [width] [height] [samples]
  ```
- **`kernel_validate`** — correctness harness that renders/evaluates the
  same scenes and BSDF/probe/texture test cases on both the CPU and the
  Vulkan/Slang backend and reports the per-pixel/per-case error, used to
  validate the Slang kernel port against the CPU reference implementation.

## Requirements

- CMake ≥ 3.20, C++17
- macOS on Apple Silicon (arm64) — currently the only supported platform,
  since the Vulkan/Slang kernel path only ships a pinned Slang toolchain for
  macOS arm64 and targets Vulkan via MoltenVK
- Vulkan SDK / MoltenVK — required (not optional); `find_package(Vulkan
  REQUIRED)` fails the configure step if it isn't found
- OpenMP (optional) — enables multithreaded CPU rendering
- No manual Slang install needed: the Slang compiler/runtime and
  [slang-rhi](https://github.com/shader-slang/slang-rhi) are fetched and
  built automatically via CMake `FetchContent`, pinned to specific
  versions/commits

## Building

```
mkdir build && cd build
cmake ..
make
make install
```

`make install` also installs the `.slang` shader sources and the Slang/
slang-rhi runtime dylibs needed by anything linking `Mirage::Mirage`.
