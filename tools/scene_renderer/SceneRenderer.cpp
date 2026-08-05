#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <utility>
#include <filesystem>

// Mirage includes
#include "mirage/core/Scene.h"
#include "mirage/prims/Primitive.h"
#include "mirage/shaders/Material.h"
#include "mirage/shaders/TextureLoader.h"
#include "mirage/utils/CrashHandler.h"
#include "mirage/utils/Log.h"
#include "mirage/utils/MathUtils.h"
#include "mirage/utils/Util.h"
#include "mirage/camera/Camera.h"
// #include "mirage/camera/PerspectiveCamera.h"
#include "mirage/math/Vec3.h"
#include "mirage/math/Transform.h"
#include "mirage/core/Renderer.h"
#include "mirage/core/RenderSettings.h"
#include "mirage/filter/NLM.h"
#include "mirage/math/Color.h"
#include "mirage/lights/Skylight.h"

// STB Image Write for saving images
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// tinyexr for 32-bit float EXR output. Vendored as tinyexr.h plus its two
// small companion headers (exr_reader.hh, streamreader.hh - both
// self-contained, standard-library-only) from upstream syoyo/tinyexr.
// Reuse the zlib decode/compress implementations already compiled into
// stb_image.h (mirage/thirdparty, STB_IMAGE_IMPLEMENTATION defined in
// TextureLoader.cpp, linked in via the Mirage library) and
// stb_image_write.h (implementation just above) instead of pulling in
// miniz as a separate dependency.
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

using namespace Mirage;

// Forward declarations
class SceneParser;
class MaterialDefinition;
class PrimitiveDefinition;

// Helper functions
std::string trim(const std::string &str)
{
    auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); });
    auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return (start < end) ? std::string(start, end) : std::string();
}

std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(trim(token));
        }
    }

    MIRAGE_LOG_DEBUG("Debug: Split '{}' into {} tokens", s, tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        MIRAGE_LOG_DEBUG("Debug: Token {}: '{}'", i, tokens[i]);
    }

    return tokens;
}

bool startsWith(const std::string &str, const std::string &prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

// Parses a comma-separated AOV name list ("depth,normal,albedo") into an
// AovType bitmask (mirage/core/Renderer.h). Returns false (leaving *outMask
// untouched) on any unrecognized name. Shared between the --aovs CLI flag
// and a scene file's `render_settings { aovs ... }` key (see
// RenderSettingsDefinition/parseRenderSettingsProperty below) so both stay
// in sync with exactly the same recognized names.
bool ParseAovMask(const std::string &commaList, uint32_t *outMask)
{
    uint32_t mask = 0;
    std::istringstream stream(commaList);
    std::string name;
    while (std::getline(stream, name, ','))
    {
        name = trim(name);
        if (name == "depth")
            mask |= Mirage::kAovDepth;
        else if (name == "normal")
            mask |= Mirage::kAovNormal;
        else if (name == "primid")
            mask |= Mirage::kAovPrimId;
        else if (name == "albedo")
            mask |= Mirage::kAovAlbedo;
        else
            return false;
    }
    *outMask = mask;
    return true;
}

// Parses a --view-transform-style name ("filmic"/"aces"/"srgb"/"none") into
// a ViewTransform (mirage/utils/Util.h). Returns false (leaving *outVt
// untouched) on any unrecognized name. Same sharing rationale as
// ParseAovMask above.
bool ParseViewTransformName(const std::string &value, Mirage::ViewTransform *outVt)
{
    if (value == "filmic")
        *outVt = Mirage::ViewTransform::eFilmic;
    else if (value == "aces")
        *outVt = Mirage::ViewTransform::eAcesLike;
    else if (value == "srgb")
        *outVt = Mirage::ViewTransform::eSrgbDisplay;
    else if (value == "none")
        *outVt = Mirage::ViewTransform::eNone;
    else
        return false;
    return true;
}

// Loads a UDIM tile set into one composited atlas Texture (Tier-2 "UDIM") -
// see TextureSampling.h's UdimAtlasUV / Material.slang's mirror for how the
// atlas is sampled once loaded, and that comment's note on this loader's
// tile-placement convention (tile (0,0) = the atlas's top-left cell in
// memory, u right, v down - no vertical flip needed relative to the
// sampler). `patternPath` must contain the literal token "<UDIM>" exactly
// once (already resolved to an absolute/baseDir-relative path by the
// caller); every file in patternPath's directory matching
// "<prefix><digits><suffix>" is treated as one tile, keyed by the standard
// UDIM numbering (1001 + u + 10*v). Returns nullptr (logging via stderr,
// same "missing/broken texture doesn't fail the whole scene load"
// convention as LoadTextureFromFile) if no tiles are found or tiles
// disagree on resolution - a partial/malformed UDIM set shouldn't silently
// misrender.
std::unique_ptr<Texture> LoadUdimAtlas(const std::string &patternPath, TextureColorSpace space)
{
    const std::string kToken = "<UDIM>";
    size_t tokenPos = patternPath.find(kToken);
    if (tokenPos == std::string::npos)
        return nullptr;

    std::string prefix = patternPath.substr(0, tokenPos);
    std::string suffix = patternPath.substr(tokenPos + kToken.size());

    std::filesystem::path dir = std::filesystem::path(prefix).parent_path();
    std::string filePrefix = std::filesystem::path(prefix).filename().string();
    if (dir.empty())
        dir = ".";

    struct Tile
    {
        int u, v;
        std::unique_ptr<Texture> tex;
    };
    std::vector<Tile> tiles;

    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec))
    {
        if (ec || !entry.is_regular_file())
            continue;

        std::string name = entry.path().filename().string();
        if (name.size() <= filePrefix.size() + suffix.size())
            continue;
        if (name.compare(0, filePrefix.size(), filePrefix) != 0)
            continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;

        std::string digits = name.substr(filePrefix.size(), name.size() - filePrefix.size() - suffix.size());
        if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); }))
            continue;

        int udim = std::stoi(digits);
        int u = (udim - 1001) % 10;
        int v = (udim - 1001) / 10;
        if (u < 0 || v < 0)
        {
            MIRAGE_LOG_ERROR("LoadUdimAtlas: skipping tile with out-of-range UDIM number {} ({})", udim, name);
            continue;
        }

        // generateMips=false - mips are generated once for the whole
        // composited atlas below (see the plan's note that per-tile mip
        // generation would need to happen twice - once per tile, once for
        // the atlas - to no benefit, since only the atlas's own mip chain
        // is ever sampled).
        auto tex = LoadTextureFromFile(entry.path().string(), space, /*generateMips=*/false);
        if (!tex)
            continue;

        tiles.push_back(Tile{u, v, std::move(tex)});
    }

    if (tiles.empty())
    {
        MIRAGE_LOG_ERROR("LoadUdimAtlas: no tiles found matching pattern '{}'", patternPath);
        return nullptr;
    }

    int tileW = tiles[0].tex->width;
    int tileH = tiles[0].tex->height;
    int gridW = 0, gridH = 0;
    for (const auto &t : tiles)
    {
        gridW = std::max(gridW, t.u + 1);
        gridH = std::max(gridH, t.v + 1);
        if (t.tex->width != tileW || t.tex->height != tileH)
        {
            MIRAGE_LOG_ERROR("LoadUdimAtlas: tile resolution mismatch (expected {}x{}, got {}x{}) - rejecting UDIM set '{}'", tileW, tileH, t.tex->width, t.tex->height, patternPath);
            return nullptr;
        }
    }

    auto atlas = std::make_unique<Texture>();
    atlas->width = gridW * tileW;
    atlas->height = gridH * tileH;
    atlas->depth = 1;
    atlas->udimGridWidth = gridW;
    atlas->udimGridHeight = gridH;
    size_t atlasTexelCount = (size_t)atlas->width * (size_t)atlas->height * 4;
    atlas->data = new float[atlasTexelCount];
    // Unpopulated cells (a sparse UDIM set, e.g. tiles 1001 and 1050 with
    // nothing authored in between) stay black rather than garbage.
    std::fill(atlas->data, atlas->data + atlasTexelCount, 0.0f);

    for (const auto &t : tiles)
    {
        int originX = t.u * tileW;
        int originY = t.v * tileH;
        for (int y = 0; y < tileH; ++y)
        {
            const float *srcRow = t.tex->data + (size_t)y * tileW * 4;
            float *dstRow = atlas->data + ((size_t)(originY + y) * atlas->width + originX) * 4;
            std::copy(srcRow, srcRow + (size_t)tileW * 4, dstRow);
        }
    }

    // Known artifact, accepted per the plan: box-filtering across the whole
    // atlas doesn't know about tile boundaries, so coarse mip levels
    // visibly bleed neighboring tiles' edge texels together. A per-tile-
    // padded atlas layout would fix this but is out of scope here.
    atlas->GenerateMips();
    return atlas;
}

// Material definition class
class MaterialDefinition
{
public:
    MaterialDefinition() {}

    void setColor(float r, float g, float b)
    {
        color = Vec3(r, g, b);
    }

    void setRoughness(float value)
    {
        roughness = value;
    }

    void setMetallic(float value)
    {
        metallic = value;
    }

    void setSpecular(float value)
    {
        specular = value;
    }

    void setEmission(float r, float g, float b)
    {
        emission = Vec3(r, g, b);
    }

    void setOpacity(float value)
    {
        opacity = value;
    }

    void setAlbedoMapPath(const std::string &path) { albedoMapPath = path; }
    void setRoughnessMapPath(const std::string &path) { roughnessMapPath = path; }
    void setMetallicMapPath(const std::string &path) { metallicMapPath = path; }
    void setOpacityMapPath(const std::string &path) { opacityMapPath = path; }
    void setNormalMapPath(const std::string &path) { normalMapPath = path; }

    const std::string &getAlbedoMapPath() const { return albedoMapPath; }
    const std::string &getRoughnessMapPath() const { return roughnessMapPath; }
    const std::string &getMetallicMapPath() const { return metallicMapPath; }
    const std::string &getOpacityMapPath() const { return opacityMapPath; }
    const std::string &getNormalMapPath() const { return normalMapPath; }

    // Texture loading/registration is deliberately not done here - it needs a
    // Scene& (for FindOrAddTexture's dedup) that this class doesn't have, and
    // the .tin file's own directory (for relative path resolution) that only
    // the parser knows. See PrimitiveDefinition::toPrimitive(), which does
    // that resolution and sets Material::xxxTextureIndex after calling this.
    Material toMaterial() const
    {
        MIRAGE_LOG_DEBUG("Debug: Converting to Material");
        Material mat;

        // Set material properties
        mat.color = color;
        MIRAGE_LOG_DEBUG("Debug: Setting color: ({}, {}, {})", mat.color.x, mat.color.y, mat.color.z);

        mat.roughness = roughness;
        MIRAGE_LOG_DEBUG("Debug: Setting roughness: {}", mat.roughness);

        mat.metallic = metallic;
        MIRAGE_LOG_DEBUG("Debug: Setting metallic: {}", mat.metallic);

        mat.specular = specular;
        MIRAGE_LOG_DEBUG("Debug: Setting specular: {}", mat.specular);

        mat.emission = emission;
        MIRAGE_LOG_DEBUG("Debug: Setting emission: ({}, {}, {})", mat.emission.x, mat.emission.y, mat.emission.z);

        mat.opacity = opacity;
        MIRAGE_LOG_DEBUG("Debug: Setting opacity: {}", mat.opacity);

        return mat;
    }

private:
    Vec3 color = Vec3(0.82f, 0.67f, 0.16f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float specular = 0.5f;
    Vec3 emission = Vec3(0.0f);
    float opacity = 1.0f;
    std::string albedoMapPath;
    std::string roughnessMapPath;
    std::string metallicMapPath;
    std::string opacityMapPath;
    std::string normalMapPath;
};

// Camera definition class - mirrors MaterialDefinition's shape (plain
// setters + a toCamera() converter). Unlike materials/primitives, a scene
// has at most one camera, so the parser stores a single optional instance
// rather than a map/vector - see SceneParser::hasCamera below.
class CameraDefinition
{
public:
    CameraDefinition() {}

    void setPosition(float x, float y, float z) { position = Vec3(x, y, z); }
    void setRotation(float x, float y, float z, float w) { rotation = Quat(x, y, z, w); }
    void setFov(float degrees) { fov = degrees; }
    void setFocalLength(float mm) { focalLength = mm; }
    void setSensorWidth(float mm) { sensorWidth = mm; }
    void setSensorHeight(float mm) { sensorHeight = mm; }
    void setFStop(float value) { fStop = value; }
    void setShutterSpeed(float seconds) { shutterSpeed = seconds; }
    void setIso(float value) { iso = value; }
    void setAperture(float value) { aperture = value; }
    void setFocalPoint(float value) { focalPoint = value; }

    Camera toCamera() const
    {
        Camera cam;
        cam.position = position;
        cam.rotation = rotation;
        cam.fov = DegToRad(fov);
        cam.focalLength = focalLength;
        cam.sensorWidth = sensorWidth;
        cam.sensorHeight = sensorHeight;
        cam.fStop = fStop;
        cam.shutterSpeed = shutterSpeed;
        cam.iso = iso;
        cam.aperture = aperture;
        cam.focalPoint = focalPoint;
        return cam;
    }

private:
    Vec3 position = Vec3(0.0f, 2.0f, 5.0f);
    Quat rotation;
    float fov = 45.0f; // degrees; converted to radians in toCamera()
    float focalLength = 0.0f;
    float sensorWidth = 36.0f;
    float sensorHeight = 24.0f;
    float fStop = 0.0f;
    float shutterSpeed = 0.0f;
    float iso = 100.0f;
    float aperture = 0.0f;
    float focalPoint = 1.0f;
};

// Punctual (point/directional) light definition class - mirrors
// CameraDefinition's shape (plain setters + a to*() converter). Unlike
// camera (at most one per scene), a scene can have any number of lights -
// see SceneParser::lights below.
class LightDefinition
{
public:
    LightDefinition() {}

    void setType(const std::string &t) { type = t; }
    void setPosition(float x, float y, float z) { position = Vec3(x, y, z); }
    void setDirection(float x, float y, float z) { direction = Vec3(x, y, z); }
    void setColor(float r, float g, float b) { color = Vec3(r, g, b); }
    void setIntensity(float value) { intensity = value; }
    void setRadius(float value) { radius = value; }
    void setAngle(float value) { angle = value; }

    PunctualLight toLight() const
    {
        PunctualLight light;
        light.type = (type == "directional") ? PunctualLightType::eDirectional : PunctualLightType::ePoint;
        light.position = position;
        light.direction = direction;
        light.color = color;
        light.intensity = intensity;
        light.radius = radius;
        light.angle = angle;
        return light;
    }

private:
    std::string type = "point";
    Vec3 position = Vec3(0.0f);
    Vec3 direction = Vec3(0.0f, -1.0f, 0.0f);
    Vec3 color = Vec3(1.0f);
    float intensity = 1.0f;
    float radius = 0.0f;
    float angle = 0.0f;
};

// Sky definition class - mirrors CameraDefinition's shape (at most one sky
// per scene, plain setters, applied directly against Scene rather than via
// a pure to*() converter since baking a Preetham sky needs to allocate a
// Probe, not just fill a POD struct). "gradient" (the default, matching
// Scene::Sky's own pre-existing defaults) sets the simple two-color
// horizon/zenith gradient; "preetham" bakes the analytic model into an HDR
// probe once at scene-load time - see Skylight.h's BakePreethamSky.
class SkyDefinition
{
public:
    SkyDefinition() {}

    void setType(const std::string &t) { type = t; }
    void setSunDirection(float x, float y, float z) { sunDirection = Vec3(x, y, z); }
    void setTurbidity(float value) { turbidity = value; }
    void setResolution(int w, int h) { resWidth = w; resHeight = h; }
    void setHorizon(float r, float g, float b) { horizon = Vec3(r, g, b); }
    void setZenith(float r, float g, float b) { zenith = Vec3(r, g, b); }

    void applyTo(Scene &scene) const
    {
        if (type == "preetham")
        {
            MIRAGE_LOG_DEBUG("Debug: Baking Preetham sky (turbidity={}, resolution={}x{})", turbidity, resWidth, resHeight);
            scene.sky.probe = BakePreethamSky(sunDirection, turbidity, resWidth, resHeight);
        }
        else
        {
            MIRAGE_LOG_DEBUG("Debug: Setting gradient sky");
            scene.sky.horizon = horizon;
            scene.sky.zenith = zenith;
        }
    }

private:
    std::string type = "gradient";
    Vec3 sunDirection = Vec3(0.3f, 0.8f, 0.2f);
    float turbidity = 3.0f;
    int resWidth = 256;
    int resHeight = 128;
    // Defaults match Scene::Sky's own pre-existing (Sky()-constructed)
    // gradient - a `sky { type gradient }` block with no horizon/zenith
    // properties set is a no-op relative to having no sky block at all.
    Vec3 horizon = Vec3(0.0f);
    Vec3 zenith = Vec3(0.0f);
};

// Primitive definition class
class PrimitiveDefinition
{
public:
    PrimitiveDefinition() : type("sphere"), materialName("default") {}

    void setType(const std::string &t)
    {
        type = t;
    }

    std::string getType() const
    {
        return type;
    }

    void setMaterialName(const std::string &name)
    {
        materialName = name;
    }

    void setPosition(float x, float y, float z)
    {
        position = Vec3(x, y, z);
    }

    void setRotation(float x, float y, float z, float w)
    {
        rotation = Quat(x, y, z, w);
    }

    void setScale(float s)
    {
        scale = s;
    }

    void setRadius(float r)
    {
        radius = r;
    }

    void setPlane(float a, float b, float c, float d)
    {
        plane[0] = a;
        plane[1] = b;
        plane[2] = c;
        plane[3] = d;
    }

    void setWidth(float w) { width = w; }
    void setHeight(float h) { height = h; }

    Primitive toPrimitive(const std::map<std::string, MaterialDefinition> &materials, Scene &scene, const std::string &baseDir) const
    {
        MIRAGE_LOG_DEBUG("Debug: Creating primitive of type: {}", type);

        Primitive prim;

        // Set transform
        MIRAGE_LOG_DEBUG("Debug: Setting transform: position({}, {}, {})", position.x, position.y, position.z);
        prim.startTransform.p = position;
        prim.startTransform.r = rotation;
        prim.startTransform.s = scale;

        prim.endTransform = prim.startTransform;

        // Set type and geometry
        if (type == "sphere") {
            MIRAGE_LOG_DEBUG("Debug: Setting sphere with radius: {}", radius);
            prim.type = Type::eSphere;
            prim.sphere.radius = radius;
        }
        else if (type == "plane") {
            MIRAGE_LOG_DEBUG("Debug: Setting plane with normal: ({}, {}, {}, {})", plane[0], plane[1], plane[2], plane[3]);
            prim.type = Type::ePlane;
            prim.plane.plane[0] = plane[0];
            prim.plane.plane[1] = plane[1];
            prim.plane.plane[2] = plane[2];
            prim.plane.plane[3] = plane[3];
        }
        else if (type == "rect") {
            MIRAGE_LOG_DEBUG("Debug: Setting rect with width/height: {}, {}", width, height);
            prim.type = Type::eRect;
            prim.rect.width = width;
            prim.rect.height = height;
        }
        else if (type == "disk") {
            MIRAGE_LOG_DEBUG("Debug: Setting disk with radius: {}", radius);
            prim.type = Type::eDisk;
            prim.disk.radius = radius;
        }

        // Set material
        MIRAGE_LOG_DEBUG("Debug: Setting material: {}", materialName);
        Material mat;
        auto it = materials.find(materialName);
        if (it != materials.end()) {
            mat = it->second.toMaterial();
            MIRAGE_LOG_DEBUG("Debug: Material found and set");

            // Resolve + load texture maps, relative to the .tin file's own
            // directory (not the process CWD). A missing/broken texture file
            // logs via LoadTextureFromFile's own stderr message and falls
            // back to the material's scalar/color field (xxxTextureIndex
            // stays -1) rather than failing the whole scene load.
            auto resolveAndLoad = [&](const std::string &relPath, TextureColorSpace space) -> int {
                if (relPath.empty())
                    return -1;
                std::string resolvedPath = baseDir.empty() ? relPath : (std::filesystem::path(baseDir) / relPath).string();

                // A literal "<UDIM>" token dispatches to the atlas loader
                // instead of a plain single-file load - every other
                // (non-UDIM) material map path is completely unaffected by
                // this check.
                if (resolvedPath.find("<UDIM>") != std::string::npos)
                {
                    auto atlas = LoadUdimAtlas(resolvedPath, space);
                    if (!atlas)
                        return -1;
                    return scene.FindOrAddTexture(resolvedPath, std::move(atlas));
                }

                auto tex = LoadTextureFromFile(resolvedPath, space);
                if (!tex)
                    return -1;
                return scene.FindOrAddTexture(resolvedPath, std::move(tex));
            };

            mat.albedoTextureIndex = resolveAndLoad(it->second.getAlbedoMapPath(), TextureColorSpace::eSRGB);
            mat.roughnessTextureIndex = resolveAndLoad(it->second.getRoughnessMapPath(), TextureColorSpace::eLinear);
            mat.metallicTextureIndex = resolveAndLoad(it->second.getMetallicMapPath(), TextureColorSpace::eLinear);
            // Linear, not sRGB - opacityMap's alpha channel is a coverage
            // mask, not a color, same reasoning as roughness/metallic above.
            mat.opacityTextureIndex = resolveAndLoad(it->second.getOpacityMapPath(), TextureColorSpace::eLinear);
            // Linear, not sRGB - a normal map's RGB channels encode a
            // tangent-space direction, not a color, same reasoning as
            // roughness/metallic/opacity above.
            mat.normalTextureIndex = resolveAndLoad(it->second.getNormalMapPath(), TextureColorSpace::eLinear);
        }
        else {
            MIRAGE_LOG_DEBUG("Debug: Material not found, using default");
        }
        // FindOrAddMaterial dedups by name, so every primitive referencing the
        // same material block shares one Scene::materials entry rather than
        // getting its own copy.
        prim.materialIndex = scene.FindOrAddMaterial(materialName, std::make_unique<Material>(mat));

        // Set light samples if it's a light
        if (mat.emission.x > 0 ||
            mat.emission.y > 0 ||
            mat.emission.z > 0) {
            MIRAGE_LOG_DEBUG("Debug: Setting as light with samples: 1");
            prim.lightSamples = 1;
        }

        return prim;
    }

private:
    std::string type;
    std::string materialName;
    Vec3 position = Vec3(0.0f);
    Mirage::Quat rotation;
    float scale = 1.0f;
    float radius = 1.0f;
    float plane[4] = {0.0f, 1.0f, 0.0f, 0.0f}; // Default: y-up plane at origin
    float width = 1.0f;  // eRect only
    float height = 1.0f; // eRect only
};

// One `render_settings { }` scene-file block (docs/PRODUCTION_READINESS.md's
// Tier 3 "no render-settings-map / multi-product batch pipeline" finding) -
// mirrors CameraDefinition's shape (plain setters, defaults matching the
// pre-existing CLI-flag defaults). Multiple render_settings blocks are
// grouped into mirage/core/RenderSettings.h's RenderPass/RenderProduct model
// by BuildRenderSettingsFromDefs() in main() below, not here - a single
// definition doesn't know which other definitions it shares a pass with.
class RenderSettingsDefinition
{
public:
    RenderSettingsDefinition() {}

    void setName(const std::string &value) { name = value; }
    void setOutput(const std::string &value) { output = value; }
    void setAovs(const std::string &value) { aovsRaw = value; }
    void setViewTransform(const std::string &value) { viewTransformRaw = value; }
    // -1 sentinel = "not set in this block" - BuildRenderSettingsFromDefs
    // falls back to the CLI positional width/height/samples for any block
    // that omits them, same as a scene file with no render_settings blocks
    // at all.
    void setSamples(int value) { samples = value; }
    void setWidth(int value) { width = value; }
    void setHeight(int value) { height = value; }

    std::string name = "beauty";
    std::string output;
    std::string aovsRaw;                          // raw comma-list, "" = beauty only (aovMask 0)
    std::string viewTransformRaw = "filmic";
    int samples = -1;
    int width = -1;
    int height = -1;
};

// Scene parser class
class SceneParser
{
public:
    SceneParser()
    {
        // Add a default material
        materials["default"] = MaterialDefinition();
    }

    bool parse(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            MIRAGE_LOG_ERROR("Failed to open file: {}", filename);
            return false;
        }

        baseDir = std::filesystem::path(filename).parent_path().string();

        MIRAGE_LOG_DEBUG("Debug: File opened successfully: {}", filename);

        std::string line;
        std::string currentSection;
        std::string currentMaterial;
        PrimitiveDefinition currentPrimitive;
        CameraDefinition currentCamera;
        LightDefinition currentLight;
        SkyDefinition currentSky;
        RenderSettingsDefinition currentRenderSettings;
        bool inBlock = false;
        bool waitingForBlockStart = false;

        while (std::getline(file, line)) {
            // Trim whitespace
            line = trim(line);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            MIRAGE_LOG_DEBUG("Debug: Processing line: {}", line);

            // Check if we're waiting for a block start
            if (waitingForBlockStart) {
                if (line == "{") {
                    waitingForBlockStart = false;
                    inBlock = true;
                    MIRAGE_LOG_DEBUG("Debug: Block start found");
                    continue;
                }
            }

            // Check for section start/end
            if (line == "{") {
                // Block start
                inBlock = true;
                MIRAGE_LOG_DEBUG("Debug: Block start found");
            } 
            else if (line == "}") {
                // Block end
                inBlock = false;
                
                if (currentSection == "primitive") {
                    // Add the completed primitive to the list
                    MIRAGE_LOG_DEBUG("Debug: Adding primitive to list, type: {}", currentPrimitive.getType());
                    primitives.push_back(currentPrimitive);
                    currentPrimitive = PrimitiveDefinition();
                }
                else if (currentSection == "camera") {
                    // A scene has at most one camera - a second `camera { }`
                    // block silently overwrites the first, same "last one
                    // wins" behavior as re-declaring an existing material.
                    MIRAGE_LOG_DEBUG("Debug: Setting scene camera");
                    camera = currentCamera;
                    hasCamera = true;
                    currentCamera = CameraDefinition();
                }
                else if (currentSection == "light") {
                    MIRAGE_LOG_DEBUG("Debug: Adding light to list");
                    lights.push_back(currentLight);
                    currentLight = LightDefinition();
                }
                else if (currentSection == "sky") {
                    // A scene has at most one sky - a second `sky { }` block
                    // silently overwrites the first, same "last one wins"
                    // behavior as camera.
                    MIRAGE_LOG_DEBUG("Debug: Setting scene sky");
                    sky = currentSky;
                    hasSky = true;
                    currentSky = SkyDefinition();
                }
                else if (currentSection == "render_settings") {
                    // Repeatable, like primitive - each block is one output
                    // product (see RenderSettingsDefinition/main() below).
                    MIRAGE_LOG_DEBUG("Debug: Adding render_settings block to list");
                    renderSettingsDefs.push_back(currentRenderSettings);
                    currentRenderSettings = RenderSettingsDefinition();
                }

                currentSection = "";
            } 
            else if (!inBlock && !waitingForBlockStart) {
                // Check if this is a section declaration
                std::vector<std::string> parts = split(line, ' ');
                if (parts.size() >= 1) {
                    if (parts[0] == "material" && parts.size() >= 2) {
                        currentSection = "material";
                        currentMaterial = parts[1];
                        MIRAGE_LOG_DEBUG("Debug: Starting material section: {}", currentMaterial);
                        
                        // Initialize material with defaults
                        materials[currentMaterial] = MaterialDefinition();
                        waitingForBlockStart = true;
                    } 
                    else if (parts[0] == "primitive") {
                        currentSection = "primitive";
                        MIRAGE_LOG_DEBUG("Debug: Starting primitive section");

                        // Initialize new primitive
                        currentPrimitive = PrimitiveDefinition();
                        waitingForBlockStart = true;
                    }
                    else if (parts[0] == "camera") {
                        currentSection = "camera";
                        MIRAGE_LOG_DEBUG("Debug: Starting camera section");

                        currentCamera = CameraDefinition();
                        waitingForBlockStart = true;
                    }
                    else if (parts[0] == "light") {
                        currentSection = "light";
                        MIRAGE_LOG_DEBUG("Debug: Starting light section");

                        currentLight = LightDefinition();
                        waitingForBlockStart = true;
                    }
                    else if (parts[0] == "sky") {
                        currentSection = "sky";
                        MIRAGE_LOG_DEBUG("Debug: Starting sky section");

                        currentSky = SkyDefinition();
                        waitingForBlockStart = true;
                    }
                    else if (parts[0] == "render_settings") {
                        currentSection = "render_settings";
                        MIRAGE_LOG_DEBUG("Debug: Starting render_settings section");

                        currentRenderSettings = RenderSettingsDefinition();
                        waitingForBlockStart = true;
                    }
                }
            }
            else if (inBlock) {
                // Parse property within a section
                std::vector<std::string> parts = split(line, ' ');

                if (currentSection == "material") {
                    parseMaterialProperty(currentMaterial, parts);
                }
                else if (currentSection == "primitive") {
                    parsePrimitiveProperty(currentPrimitive, parts);
                }
                else if (currentSection == "camera") {
                    parseCameraProperty(currentCamera, parts);
                }
                else if (currentSection == "light") {
                    parseLightProperty(currentLight, parts);
                }
                else if (currentSection == "sky") {
                    parseSkyProperty(currentSky, parts);
                }
                else if (currentSection == "render_settings") {
                    parseRenderSettingsProperty(currentRenderSettings, parts);
                }
            }
        }

        // If we're still in a block at the end of the file, handle it
        if (inBlock && currentSection == "primitive") {
            MIRAGE_LOG_DEBUG("Debug: Adding final primitive to list");
            primitives.push_back(currentPrimitive);
        }
        else if (inBlock && currentSection == "camera") {
            MIRAGE_LOG_DEBUG("Debug: Setting final scene camera");
            camera = currentCamera;
            hasCamera = true;
        }
        else if (inBlock && currentSection == "light") {
            MIRAGE_LOG_DEBUG("Debug: Adding final light to list");
            lights.push_back(currentLight);
        }
        else if (inBlock && currentSection == "sky") {
            MIRAGE_LOG_DEBUG("Debug: Setting final scene sky");
            sky = currentSky;
            hasSky = true;
        }
        else if (inBlock && currentSection == "render_settings") {
            MIRAGE_LOG_DEBUG("Debug: Adding final render_settings block to list");
            renderSettingsDefs.push_back(currentRenderSettings);
        }


        MIRAGE_LOG_DEBUG("Debug: Parsing complete. Found {} materials and {} primitives", materials.size(), primitives.size());
        
        return true;
    }

    void setupScene(Scene &scene)
    {
        // Clear existing scene
        MIRAGE_LOG_DEBUG("Debug: Clearing existing scene");
        scene.Clear();

        // Add primitives
        MIRAGE_LOG_DEBUG("Debug: Adding {} primitives to scene", primitives.size());
        for (const auto &primDef : primitives)
        {
            try
            {
                Primitive prim = primDef.toPrimitive(materials, scene, baseDir);
                MIRAGE_LOG_DEBUG("Debug: Adding primitive of type {}", primDef.getType());
                scene.AddPrimitive(prim);
            }
            catch (const std::exception &e)
            {
                MIRAGE_LOG_ERROR("Exception while adding primitive: {}", e.what());
            }
            catch (...)
            {
                MIRAGE_LOG_ERROR("Unknown exception while adding primitive");
            }
        }

        // Add punctual (point/directional) lights.
        MIRAGE_LOG_DEBUG("Debug: Adding {} lights to scene", lights.size());
        for (const auto &lightDef : lights)
        {
            scene.lights.push_back(lightDef.toLight());
        }

        // Apply the parsed sky, if a `sky { }` block was present. If not,
        // scene.sky stays at Scene::Clear()'s Sky() defaults (the same
        // simple all-zero gradient as before this feature existed).
        if (hasSky)
        {
            MIRAGE_LOG_DEBUG("Debug: Applying parsed sky to scene");
            sky.applyTo(scene);
        }

        // Add the parsed camera, if a `camera { }` block was present. If not,
        // scene.camera stays null here - main() falls back to its own
        // hardcoded default camera in that case, so scenes without a camera
        // block behave exactly as before this feature existed.
        if (hasCamera)
        {
            MIRAGE_LOG_DEBUG("Debug: Adding parsed camera to scene");
            scene.camera = std::make_unique<Camera>(camera.toCamera());
        }

        // Build the scene (BVH, etc.)
        MIRAGE_LOG_DEBUG("Debug: Building scene (BVH)");
        try
        {
            scene.Build();
            MIRAGE_LOG_DEBUG("Debug: Scene built successfully");
        }
        catch (const std::exception &e)
        {
            MIRAGE_LOG_ERROR("Exception while building scene: {}", e.what());
        }
        catch (...)
        {
            MIRAGE_LOG_ERROR("Unknown exception while building scene");
        }
    }

    // The scene file's parsed render_settings { } blocks (possibly empty -
    // most scene files have none, see main()'s fallback to
    // MakeLegacyRenderSettings when this is empty).
    const std::vector<RenderSettingsDefinition> &GetRenderSettingsDefs() const { return renderSettingsDefs; }

private:
    void parseMaterialProperty(const std::string &materialName, const std::vector<std::string> &parts)
    {
        if (parts.empty()) return;

        std::string property = parts[0];
        MIRAGE_LOG_DEBUG("Debug: Parsing material property: {} for material: {}", property, materialName);

        try {
            if (property == "color" && parts.size() >= 4) {
                float r = std::stof(parts[1]);
                float g = std::stof(parts[2]);
                float b = std::stof(parts[3]);
                MIRAGE_LOG_DEBUG("Debug: Setting color: ({}, {}, {})", r, g, b);
                materials[materialName].setColor(r, g, b);
            }
            else if (property == "roughness" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting roughness: {}", value);
                materials[materialName].setRoughness(value);
            }
            else if (property == "metallic" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting metallic: {}", value);
                materials[materialName].setMetallic(value);
            }
            else if (property == "specular" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting specular: {}", value);
                materials[materialName].setSpecular(value);
            }
            else if (property == "emission" && parts.size() >= 4) {
                float r = std::stof(parts[1]);
                float g = std::stof(parts[2]);
                float b = std::stof(parts[3]);
                MIRAGE_LOG_DEBUG("Debug: Setting emission: ({}, {}, {})", r, g, b);
                materials[materialName].setEmission(r, g, b);
            }
            else if (property == "opacity" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting opacity: {}", value);
                materials[materialName].setOpacity(value);
            }
            // Paths with spaces aren't supported - this parser's tokenizer
            // (split()) splits on any space, same pre-existing limitation as
            // every other multi-token property above.
            else if (property == "albedoMap" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting albedoMap: {}", parts[1]);
                materials[materialName].setAlbedoMapPath(parts[1]);
            }
            else if (property == "roughnessMap" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting roughnessMap: {}", parts[1]);
                materials[materialName].setRoughnessMapPath(parts[1]);
            }
            else if (property == "metallicMap" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting metallicMap: {}", parts[1]);
                materials[materialName].setMetallicMapPath(parts[1]);
            }
            else if (property == "opacityMap" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting opacityMap: {}", parts[1]);
                materials[materialName].setOpacityMapPath(parts[1]);
            }
            else if (property == "normalMap" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting normalMap: {}", parts[1]);
                materials[materialName].setNormalMapPath(parts[1]);
            }
        }
        catch (const std::exception& e) {
            MIRAGE_LOG_ERROR("Exception while parsing material property: {}", e.what());
        }
    }

    void parsePrimitiveProperty(PrimitiveDefinition& primitive, const std::vector<std::string>& parts) {
        if (parts.empty()) return;

        std::string property = parts[0];
        MIRAGE_LOG_DEBUG("Debug: Parsing primitive property: {}", property);

        try {
            if (property == "type" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting primitive type: {}", parts[1]);
                primitive.setType(parts[1]);
            }
            else if (property == "material" && parts.size() >= 2) {
                MIRAGE_LOG_DEBUG("Debug: Setting primitive material: {}", parts[1]);
                primitive.setMaterialName(parts[1]);
            }
            else if (property == "position" && parts.size() >= 4) {
                float x = std::stof(parts[1]);
                float y = std::stof(parts[2]);
                float z = std::stof(parts[3]);
                MIRAGE_LOG_DEBUG("Debug: Setting position: ({}, {}, {})", x, y, z);
                primitive.setPosition(x, y, z);
            }
            else if (property == "rotation" && parts.size() >= 5) {
                float x = std::stof(parts[1]);
                float y = std::stof(parts[2]);
                float z = std::stof(parts[3]);
                float w = std::stof(parts[4]);
                MIRAGE_LOG_DEBUG("Debug: Setting rotation: ({}, {}, {}, {})", x, y, z, w);
                primitive.setRotation(x, y, z, w);
            }
            else if (property == "scale" && parts.size() >= 2) {
                float s = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting scale: {}", s);
                primitive.setScale(s);
            }
            else if (property == "radius" && parts.size() >= 2) {
                float r = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting radius: {}", r);
                primitive.setRadius(r);
            }
            else if (property == "plane" && parts.size() >= 5) {
                float a = std::stof(parts[1]);
                float b = std::stof(parts[2]);
                float c = std::stof(parts[3]);
                float d = std::stof(parts[4]);
                MIRAGE_LOG_DEBUG("Debug: Setting plane: ({}, {}, {}, {})", a, b, c, d);
                primitive.setPlane(a, b, c, d);
            }
            else if (property == "width" && parts.size() >= 2) {
                float w = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting width: {}", w);
                primitive.setWidth(w);
            }
            else if (property == "height" && parts.size() >= 2) {
                float h = std::stof(parts[1]);
                MIRAGE_LOG_DEBUG("Debug: Setting height: {}", h);
                primitive.setHeight(h);
            }
        }
        catch (const std::exception& e) {
            MIRAGE_LOG_ERROR("Exception while parsing primitive property: {}", e.what());
        }
    }

    void parseCameraProperty(CameraDefinition &cam, const std::vector<std::string> &parts) {
        if (parts.empty()) return;

        std::string property = parts[0];
        MIRAGE_LOG_DEBUG("Debug: Parsing camera property: {}", property);

        try {
            if (property == "position" && parts.size() >= 4) {
                cam.setPosition(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
            else if (property == "rotation" && parts.size() >= 5) {
                cam.setRotation(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]), std::stof(parts[4]));
            }
            else if (property == "fov" && parts.size() >= 2) {
                cam.setFov(std::stof(parts[1]));
            }
            else if (property == "focalLength" && parts.size() >= 2) {
                cam.setFocalLength(std::stof(parts[1]));
            }
            else if (property == "sensorWidth" && parts.size() >= 2) {
                cam.setSensorWidth(std::stof(parts[1]));
            }
            else if (property == "sensorHeight" && parts.size() >= 2) {
                cam.setSensorHeight(std::stof(parts[1]));
            }
            else if (property == "fStop" && parts.size() >= 2) {
                cam.setFStop(std::stof(parts[1]));
            }
            else if (property == "shutterSpeed" && parts.size() >= 2) {
                cam.setShutterSpeed(std::stof(parts[1]));
            }
            else if (property == "iso" && parts.size() >= 2) {
                cam.setIso(std::stof(parts[1]));
            }
            else if (property == "aperture" && parts.size() >= 2) {
                cam.setAperture(std::stof(parts[1]));
            }
            else if (property == "focalPoint" && parts.size() >= 2) {
                cam.setFocalPoint(std::stof(parts[1]));
            }
        }
        catch (const std::exception& e) {
            MIRAGE_LOG_ERROR("Exception while parsing camera property: {}", e.what());
        }
    }

    void parseLightProperty(LightDefinition &light, const std::vector<std::string> &parts) {
        if (parts.empty()) return;

        std::string property = parts[0];
        MIRAGE_LOG_DEBUG("Debug: Parsing light property: {}", property);

        try {
            if (property == "type" && parts.size() >= 2) {
                light.setType(parts[1]);
            }
            else if (property == "position" && parts.size() >= 4) {
                light.setPosition(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
            else if (property == "direction" && parts.size() >= 4) {
                light.setDirection(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
            else if (property == "color" && parts.size() >= 4) {
                light.setColor(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
            else if (property == "intensity" && parts.size() >= 2) {
                light.setIntensity(std::stof(parts[1]));
            }
            else if (property == "radius" && parts.size() >= 2) {
                light.setRadius(std::stof(parts[1]));
            }
            else if (property == "angle" && parts.size() >= 2) {
                light.setAngle(std::stof(parts[1]));
            }
        }
        catch (const std::exception& e) {
            MIRAGE_LOG_ERROR("Exception while parsing light property: {}", e.what());
        }
    }

    void parseSkyProperty(SkyDefinition &s, const std::vector<std::string> &parts) {
        if (parts.empty()) return;

        std::string property = parts[0];
        MIRAGE_LOG_DEBUG("Debug: Parsing sky property: {}", property);

        try {
            if (property == "type" && parts.size() >= 2) {
                s.setType(parts[1]);
            }
            else if (property == "sunDirection" && parts.size() >= 4) {
                s.setSunDirection(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
            else if (property == "turbidity" && parts.size() >= 2) {
                s.setTurbidity(std::stof(parts[1]));
            }
            else if (property == "resolution" && parts.size() >= 3) {
                s.setResolution(std::stoi(parts[1]), std::stoi(parts[2]));
            }
            else if (property == "horizon" && parts.size() >= 4) {
                s.setHorizon(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
            else if (property == "zenith" && parts.size() >= 4) {
                s.setZenith(std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]));
            }
        }
        catch (const std::exception& e) {
            MIRAGE_LOG_ERROR("Exception while parsing sky property: {}", e.what());
        }
    }

    // Recognized keys: name, output, aovs <comma-list>, view_transform,
    // samples, width, height - see RenderSettingsDefinition's field comments
    // and main()'s BuildRenderSettingsFromDefs() for how these get grouped
    // into RenderPass/RenderProduct. aovs/view_transform values aren't
    // validated here (ParseAovMask/ParseViewTransformName failures are
    // reported once, at grouping time in main(), where the error message can
    // also name which render_settings block - by its `name` key - triggered
    // it).
    void parseRenderSettingsProperty(RenderSettingsDefinition &rs, const std::vector<std::string> &parts) {
        if (parts.empty()) return;

        std::string property = parts[0];
        MIRAGE_LOG_DEBUG("Debug: Parsing render_settings property: {}", property);

        try {
            if (property == "name" && parts.size() >= 2) {
                rs.setName(parts[1]);
            }
            else if (property == "output" && parts.size() >= 2) {
                rs.setOutput(parts[1]);
            }
            else if (property == "aovs" && parts.size() >= 2) {
                rs.setAovs(parts[1]);
            }
            else if (property == "view_transform" && parts.size() >= 2) {
                rs.setViewTransform(parts[1]);
            }
            else if (property == "samples" && parts.size() >= 2) {
                rs.setSamples(std::stoi(parts[1]));
            }
            else if (property == "width" && parts.size() >= 2) {
                rs.setWidth(std::stoi(parts[1]));
            }
            else if (property == "height" && parts.size() >= 2) {
                rs.setHeight(std::stoi(parts[1]));
            }
        }
        catch (const std::exception& e) {
            MIRAGE_LOG_ERROR("Exception while parsing render_settings property: {}", e.what());
        }
    }

    std::map<std::string, MaterialDefinition> materials;
    std::vector<PrimitiveDefinition> primitives;
    std::vector<LightDefinition> lights;
    CameraDefinition camera;
    bool hasCamera = false;
    SkyDefinition sky;
    bool hasSky = false;
    // Empty (the common case) means the scene file had no render_settings
    // blocks at all - main() falls back to MakeLegacyRenderSettings() using
    // the positional CLI args + --aovs/--view-transform/--denoise flags.
    std::vector<RenderSettingsDefinition> renderSettingsDefs;
    // The .tin file's own directory, set by parse() - albedoMap/roughnessMap/
    // metallicMap paths are resolved relative to this, not the process's
    // current working directory.
    std::string baseDir;
};

// Writes the raw (untonemapped, unquantized) float32 RGBA render buffer to
// an OpenEXR file. Unlike the LDR writers in main() below, this preserves
// the renderer's full dynamic range - Color is already float32 RGBA
// (mirage/math/Color.h), so no conversion beyond a component split is
// needed.
bool WriteExr(const std::string &path, const Color *pixels, int width, int height)
{
    // outputImage[i].w today holds the CPU splat buffer's accumulated filter
    // weight (see CpuRenderer::AddSample in mirage/core/Renderer.cpp), not
    // real alpha/coverage, and the GPU path's resolved .w is the raw
    // accumulated sample weight too - neither is a [0,1] coverage value, so
    // write a fully-opaque alpha channel instead of passing it through.
    std::vector<float> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    for (int i = 0; i < width * height; ++i)
    {
        rgba[i * 4 + 0] = pixels[i].x;
        rgba[i * 4 + 1] = pixels[i].y;
        rgba[i * 4 + 2] = pixels[i].z;
        rgba[i * 4 + 3] = 1.0f;
    }

    const char *err = nullptr;
    int ret = SaveEXR(rgba.data(), width, height, 4, /*save_as_fp16=*/0, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        MIRAGE_LOG_ERROR("Failed to save EXR: {}", (err ? err : "unknown error"));
        if (err)
            FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

// Multi-layer float32 EXR writer: beauty plus whichever AovType bits are set
// in `aovMask` (Tier B/NamedAov AOVs aren't wired into this writer yet - see
// Renderer.h's NamedAov comment; only the fixed AovType set is supported
// here). Uses tinyexr's full EXRHeader/EXRImage API (SaveEXR() above only
// wraps the single flat-RGBA case) with "layer.channel" channel naming -
// bare R/G/B/A for the beauty layer (no prefix, so ordinary single-layer EXR
// viewers/readers see it exactly like WriteExr's output), "depth.Z"
// (scalar), "normal.X/Y/Z", "albedo.R/G/B" for AOV layers - matching the
// convention most compositors (Nuke, Blender, etc.) expect for multi-AOV
// EXRs. Only layers whose AovType bit is set (and whose AovBuffers pointer
// is non-null) are written, so a single-AOV request produces a small
// beauty+one-layer file, not a fixed maximal channel set.
bool WriteMultiLayerExr(const std::string &path, const Color *beauty, const AovBuffers &aovs, uint32_t aovMask, int width, int height)
{
    struct Layer
    {
        std::string name; // "" for the beauty layer (no "layer." prefix)
        std::vector<std::pair<std::string, int>> channels; // {suffix, Color component index 0..3}
        const Color *buffer;
    };

    std::vector<Layer> layers;
    // (A)BGR order, matching SaveEXR()'s own convention above - "most EXR
    // viewers expect this channel order" per tinyexr's SaveEXR comment.
    layers.push_back({"", {{"A", 3}, {"B", 2}, {"G", 1}, {"R", 0}}, beauty});
    if ((aovMask & kAovDepth) && aovs.depth)
        layers.push_back({"depth", {{"Z", 0}}, aovs.depth}); // AovBuffers::depth's .x holds hit distance
    if ((aovMask & kAovNormal) && aovs.normal)
        layers.push_back({"normal", {{"Z", 2}, {"Y", 1}, {"X", 0}}, aovs.normal});
    if ((aovMask & kAovPrimId) && aovs.primId)
        layers.push_back({"primId", {{"Z", 0}}, aovs.primId}); // .x holds (float)Primitive::hydraId
    if ((aovMask & kAovAlbedo) && aovs.albedo)
        layers.push_back({"albedo", {{"B", 2}, {"G", 1}, {"R", 0}}, aovs.albedo});

    size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

    // SoA per-channel float arrays - tinyexr's EXRImage::images layout, not
    // WriteExr's interleaved RGBA (see SaveEXR()'s own "Split ... into R, G
    // and B(and A) layers" comment above for the same transform on the
    // single-layer path).
    std::vector<std::vector<float>> channelData;
    std::vector<std::string> channelNames;
    for (const Layer &layer : layers)
    {
        for (const auto &[suffix, component] : layer.channels)
        {
            std::vector<float> data(pixelCount);
            for (size_t i = 0; i < pixelCount; ++i)
                data[i] = layer.buffer[i][component];
            channelData.push_back(std::move(data));
            channelNames.push_back(layer.name.empty() ? suffix : (layer.name + "." + suffix));
        }
    }

    int numChannels = static_cast<int>(channelNames.size());

    EXRHeader header;
    InitEXRHeader(&header);
    header.compression_type = (width < 16 && height < 16) ? TINYEXR_COMPRESSIONTYPE_NONE : TINYEXR_COMPRESSIONTYPE_ZIP;
    header.num_channels = numChannels;
    header.channels = static_cast<EXRChannelInfo *>(malloc(sizeof(EXRChannelInfo) * static_cast<size_t>(numChannels)));
    header.pixel_types = static_cast<int *>(malloc(sizeof(int) * static_cast<size_t>(numChannels)));
    header.requested_pixel_types = static_cast<int *>(malloc(sizeof(int) * static_cast<size_t>(numChannels)));
    for (int i = 0; i < numChannels; ++i)
    {
        // channelNames entries are always short ("albedo.R" etc.), well
        // under EXRChannelInfo::name's 255-byte limit - no truncation risk.
        std::memset(header.channels[i].name, 0, sizeof(header.channels[i].name));
        std::strncpy(header.channels[i].name, channelNames[i].c_str(), sizeof(header.channels[i].name) - 1);
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // full float32, no half-precision downcast
    }

    std::vector<float *> imagePtrs(numChannels);
    for (int i = 0; i < numChannels; ++i)
        imagePtrs[i] = channelData[i].data();

    EXRImage image;
    InitEXRImage(&image);
    image.num_channels = numChannels;
    image.images = reinterpret_cast<unsigned char **>(imagePtrs.data());
    image.width = width;
    image.height = height;

    const char *err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS)
    {
        MIRAGE_LOG_ERROR("Failed to save multi-layer EXR: {}", (err ? err : "unknown error"));
        if (err)
            FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

// Writes one RenderProduct's file from its RenderPass's completed shared
// beauty/AOV buffers - chooses EXR (multi-layer if the product requested
// any AOVs, single-layer beauty-only otherwise) or an LDR format based on
// outputPath's extension, the same logic the pre-multi-product code had
// inline in main(). EXR output is always raw/untonemapped (product's
// viewTransform is intentionally not applied there) - matches WriteExr's
// existing documented convention of preserving full dynamic range; only LDR
// output routes through ApplyViewTransform(product.viewTransform).
bool WriteRenderProduct(const RenderProduct &product, const Color *beauty, const AovBuffers &aovs, int width, int height,
                         float exposure)
{
    // Local copy, not a reference - the "unrecognized extension" fallback
    // below rewrites it (e.g. "out.foo" -> "out.foo.png").
    std::string outputFile = product.outputPath;
    std::string extension = outputFile.substr(outputFile.find_last_of(".") + 1);

    if (extension == "exr")
    {
        if (product.aovMask != 0)
        {
            MIRAGE_LOG_DEBUG("Debug: Writing multi-layer float EXR output for product '{}' (beauty + AOVs)", product.name);
            return WriteMultiLayerExr(outputFile, beauty, aovs, product.aovMask, width, height);
        }
        MIRAGE_LOG_DEBUG("Debug: Writing float EXR output for product '{}'", product.name);
        return WriteExr(outputFile, beauty, width, height);
    }

    if (product.aovMask != 0)
    {
        // AOVs need per-channel layers, which only EXR supports here -
        // matches WriteMultiLayerExr's doc comment. Beauty still writes
        // normally below; the requested AOVs are just silently dropped
        // from a non-EXR output rather than erroring the whole render.
        MIRAGE_LOG_WARN("Product '{}' requested AOVs but output extension '{}' isn't EXR - AOVs need per-channel EXR "
                         "output, only the beauty image will be written",
                         product.name, extension);
    }

    MIRAGE_LOG_DEBUG("Debug: Converting image for saving (product '{}')", product.name);
    std::vector<unsigned char> imageData(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    for (int i = 0; i < width * height; ++i)
    {
        Color pixel = ApplyViewTransform(beauty[i], product.viewTransform, exposure);

        imageData[i * 3 + 0] = static_cast<unsigned char>(255.0f * Clamp(pixel.x, 0.0f, 1.0f));
        imageData[i * 3 + 1] = static_cast<unsigned char>(255.0f * Clamp(pixel.y, 0.0f, 1.0f));
        imageData[i * 3 + 2] = static_cast<unsigned char>(255.0f * Clamp(pixel.z, 0.0f, 1.0f));
    }

    bool success = false;
    if (extension == "png")
    {
        success = stbi_write_png(outputFile.c_str(), width, height, 3, imageData.data(), width * 3) != 0;
    }
    else if (extension == "jpg" || extension == "jpeg")
    {
        success = stbi_write_jpg(outputFile.c_str(), width, height, 3, imageData.data(), 95) != 0;
    }
    else if (extension == "bmp")
    {
        success = stbi_write_bmp(outputFile.c_str(), width, height, 3, imageData.data()) != 0;
    }
    else if (extension == "tga")
    {
        success = stbi_write_tga(outputFile.c_str(), width, height, 3, imageData.data()) != 0;
    }
    else
    {
        // Unrecognized extension: replace it with .png rather than
        // appending, so e.g. "out.foo" becomes "out.png", not "out.foo.png".
        MIRAGE_LOG_WARN("Product '{}': unrecognized output extension '{}', defaulting to PNG", product.name, extension);
        size_t dotPos = outputFile.find_last_of(".");
        outputFile = (dotPos != std::string::npos) ? outputFile.substr(0, dotPos) + ".png" : outputFile + ".png";
        success = stbi_write_png(outputFile.c_str(), width, height, 3, imageData.data(), width * 3) != 0;
    }

    if (success)
        MIRAGE_LOG_DEBUG("Image saved to {}", outputFile);
    else
        MIRAGE_LOG_ERROR("Failed to save image to {}", outputFile);

    return success;
}

// Groups a scene file's render_settings { } blocks into RenderPasses -
// definitions sharing the same effective (width, height, samples) triple
// (after resolving each block's -1 "not set" sentinels against
// fallbackWidth/fallbackHeight/fallbackSamples, the positional CLI args)
// share one RenderPass/one Render() call; a differing triple starts a new
// pass. Passes are emitted in order of first appearance, not sorted.
//
// `baseOptions` must already have every field EXCEPT width/height/
// maxSamples/aovMask set (mode/type/maxDepth/filter/exposure/clamp/
// enableDOF/accumulateAovs/enableDenoise/nlmWidth/nlmFalloff) - those three
// are filled in per-pass below from each group's resolved triple/aovMask
// union.
//
// Returns an empty RenderSettings (not a partial one) on any per-block
// problem (bad aovs/view_transform name, missing output path) - logs which
// block (by its `name` key) failed, so a scene-file typo doesn't silently
// drop just that one product from an otherwise-successful multi-product
// render; the caller should treat an empty result as a fatal parse error.
RenderSettings BuildRenderSettingsFromDefs(const std::vector<RenderSettingsDefinition> &defs, const Options &baseOptions,
                                            int fallbackWidth, int fallbackHeight, int fallbackSamples)
{
    struct ResolvedProduct
    {
        RenderProduct product;
        int width, height, samples;
    };

    std::vector<ResolvedProduct> resolved;
    resolved.reserve(defs.size());

    for (const RenderSettingsDefinition &def : defs)
    {
        if (def.output.empty())
        {
            MIRAGE_LOG_ERROR("render_settings '{}': missing required 'output' key", def.name);
            return {};
        }

        uint32_t productAovMask = 0;
        if (!def.aovsRaw.empty() && !ParseAovMask(def.aovsRaw, &productAovMask))
        {
            MIRAGE_LOG_ERROR("render_settings '{}': unrecognized name in aovs '{}'", def.name, def.aovsRaw);
            return {};
        }

        Mirage::ViewTransform productViewTransform = Mirage::ViewTransform::eFilmic;
        if (!ParseViewTransformName(def.viewTransformRaw, &productViewTransform))
        {
            MIRAGE_LOG_ERROR("render_settings '{}': unrecognized view_transform '{}'", def.name, def.viewTransformRaw);
            return {};
        }

        RenderProduct product;
        product.name = def.name;
        product.outputPath = def.output;
        product.aovMask = productAovMask;
        product.viewTransform = productViewTransform;

        ResolvedProduct rp;
        rp.product = std::move(product);
        rp.width = def.width > 0 ? def.width : fallbackWidth;
        rp.height = def.height > 0 ? def.height : fallbackHeight;
        rp.samples = def.samples > 0 ? def.samples : fallbackSamples;
        resolved.push_back(std::move(rp));
    }

    RenderSettings settings;
    for (const ResolvedProduct &rp : resolved)
    {
        // Linear scan for a matching existing pass - scene files have at
        // most a handful of render_settings blocks, so this is nowhere
        // near a hot path.
        RenderPass *pass = nullptr;
        for (RenderPass &candidate : settings)
        {
            if (candidate.options.width == rp.width && candidate.options.height == rp.height &&
                candidate.options.maxSamples == rp.samples)
            {
                pass = &candidate;
                break;
            }
        }

        if (!pass)
        {
            RenderPass newPass;
            newPass.options = baseOptions;
            newPass.options.width = rp.width;
            newPass.options.height = rp.height;
            newPass.options.maxSamples = rp.samples;
            settings.push_back(std::move(newPass));
            pass = &settings.back();
        }

        pass->options.aovMask |= rp.product.aovMask;
        pass->products.push_back(rp.product);
    }

    return settings;
}

// Main function
int main(int argc, char *argv[])
{
    // Default level so usage/parse errors below are visible even before
    // --log-level (if any) is parsed out of argv.
    Mirage::Logging::Init();
    Mirage::Logging::InstallCrashHandler();

    // No existing flag-parsing infrastructure here (this CLI has always been
    // purely positional - <scene_file> <output_image> [width] [height]
    // [samples]) so each --flag[=value] below is stripped out of argv in its
    // own pass before positional parsing, rather than threading a real
    // option parser through this whole function for a handful of flags.
    Mirage::Logging::Level logLevel = Mirage::Logging::Level::eInfo;
    Mirage::ViewTransform viewTransform = Mirage::ViewTransform::eFilmic;
    uint32_t aovMask = 0;
    bool enableDenoise = false;
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i)
    {
        std::string arg = argv[i];

        static const std::string kLogLevelEq = "--log-level=";
        if (arg.rfind(kLogLevelEq, 0) == 0 || arg == "--log-level")
        {
            std::string value;
            if (arg == "--log-level")
            {
                if (i + 1 >= argc)
                {
                    MIRAGE_LOG_ERROR("--log-level requires a value (trace|debug|info|warn|error|critical)");
                    return 1;
                }
                value = argv[++i];
            }
            else
            {
                value = arg.substr(kLogLevelEq.size());
            }

            if (!Mirage::Logging::ParseLevel(value.c_str(), &logLevel))
            {
                MIRAGE_LOG_ERROR("Unrecognized --log-level value '{}' (want trace|debug|info|warn|error|critical)", value);
                return 1;
            }
            continue;
        }

        static const std::string kViewTransformEq = "--view-transform=";
        if (arg.rfind(kViewTransformEq, 0) == 0 || arg == "--view-transform")
        {
            std::string value;
            if (arg == "--view-transform")
            {
                if (i + 1 >= argc)
                {
                    MIRAGE_LOG_ERROR("--view-transform requires a value (filmic|aces|srgb|none)");
                    return 1;
                }
                value = argv[++i];
            }
            else
            {
                value = arg.substr(kViewTransformEq.size());
            }

            if (!ParseViewTransformName(value, &viewTransform))
            {
                MIRAGE_LOG_ERROR("Unrecognized --view-transform value '{}' (want filmic|aces|srgb|none)", value);
                return 1;
            }
            continue;
        }

        // Comma-separated AOV names, e.g. --aovs depth,normal,albedo. Only
        // takes effect for .exr output (see the multi-layer EXR writer
        // below) - non-EXR output stays beauty-only, matching every other
        // AOV consumer today (only HydraCompatibility.cpp reads AovBuffers
        // at all; scene_renderer had zero AOV support before this flag).
        static const std::string kAovsEq = "--aovs=";
        if (arg.rfind(kAovsEq, 0) == 0 || arg == "--aovs")
        {
            std::string value;
            if (arg == "--aovs")
            {
                if (i + 1 >= argc)
                {
                    MIRAGE_LOG_ERROR("--aovs requires a value (comma-separated: depth,normal,primid,albedo)");
                    return 1;
                }
                value = argv[++i];
            }
            else
            {
                value = arg.substr(kAovsEq.size());
            }

            if (!ParseAovMask(value, &aovMask))
            {
                MIRAGE_LOG_ERROR("Unrecognized name in --aovs '{}' (want depth|normal|primid|albedo)", value);
                return 1;
            }
            continue;
        }

        // Post-process Non-Local-Means denoise (mirage/filter/NLM.h) applied
        // to the beauty buffer after Render() returns, before any view
        // transform/write - see the pass loop below for the guide-buffer
        // (albedo/normal) wiring.
        if (arg == "--denoise")
        {
            enableDenoise = true;
            continue;
        }

        args.push_back(arg);
    }

    if (logLevel != Mirage::Logging::Level::eInfo)
        Mirage::Logging::Init(logLevel);

    if (args.size() < 3)
    {
        MIRAGE_LOG_ERROR(
            "Usage: {} [--log-level trace|debug|info|warn|error|critical] [--view-transform filmic|aces|srgb|none] "
            "[--aovs depth,normal,primid,albedo] [--denoise] <scene_file> <output_image> [width] [height] [samples]",
            args.empty() ? "scene_renderer" : args[0]);
        return 1;
    }

    std::string sceneFile = args[1];
    std::string outputFile = args[2];

    // Default rendering parameters
    int width = 1280;
    int height = 720;
    int samples = 64;

    // Parse optional parameters
    if (args.size() > 3)
        width = std::stoi(args[3]);
    if (args.size() > 4)
        height = std::stoi(args[4]);
    if (args.size() > 5)
        samples = std::stoi(args[5]);

    MIRAGE_LOG_DEBUG("Debug: Starting to parse scene file: {}", sceneFile);

    // Parse the scene file
    SceneParser parser;
    if (!parser.parse(sceneFile))
    {
        MIRAGE_LOG_ERROR("Failed to parse scene file.");
        return 1;
    }

    MIRAGE_LOG_DEBUG("Debug: Scene file parsed successfully");

    // Create and setup the scene
    Scene scene;
    MIRAGE_LOG_DEBUG("Debug: Setting up scene");
    parser.setupScene(scene);

    MIRAGE_LOG_DEBUG("Debug: Scene setup complete, primitives count: {}", scene.primitives.size());

    // Create a default camera if none was specified in the scene
    if (!scene.camera)
    {
        MIRAGE_LOG_DEBUG("Debug: Creating default camera");
        auto camera = std::make_unique<Camera>();
        camera->position = Vec3(0.0f, 2.0f, 5.0f);
        // camera->lookAt = Vec3(0.0f, 0.0f, 0.0f);
        // camera->up = Vec3(0.0f, 1.0f, 0.0f);
        camera->fov = 45.0f;
        // camera->aspect = float(width) / float(height);

        scene.camera = std::move(camera);
    }

    // Render the scene
    MIRAGE_LOG_DEBUG("Rendering scene to {} ({}x{}, {} samples)", outputFile, width, height, samples);

    // Create renderer
    MIRAGE_LOG_DEBUG("Debug: Creating renderer");
    Renderer *renderer = CreateCpuRenderer(&scene);
    if (!renderer)
    {
        MIRAGE_LOG_ERROR("Failed to create renderer");
        return 1;
    }

    // Shared config every RenderPass this invocation executes starts from.
    // width/height/maxSamples here are the positional-CLI-arg fallback
    // values: BuildRenderSettingsFromDefs overrides them per pass from each
    // render_settings block's own (or its own fallback-resolved) values, but
    // MakeLegacyRenderSettings does NOT touch them at all - for the legacy
    // (no render_settings blocks) path, these are the actual final values
    // used, so they must be set here, not left at Options' uninitialized
    // defaults.
    MIRAGE_LOG_DEBUG("Debug: Setting up rendering options");
    Options baseOptions;
    baseOptions.width = width;
    baseOptions.height = height;
    baseOptions.maxSamples = samples;
    baseOptions.maxDepth = 5;
    baseOptions.mode = RenderMode::ePathTrace;
    baseOptions.type = RenderType::eCpu;
    baseOptions.filter = Filter(FilterType::eFilterGaussian, 1.0f, 2.0f);
    baseOptions.exposure = 1.0f;
    baseOptions.clamp = 10.0f;
    baseOptions.enableDOF = false;
    baseOptions.enableDenoise = enableDenoise;
    // NLM denoise defaults - see Renderer.h's Options::nlmWidth/nlmFalloff
    // comment. No CLI flag exposes these yet (only on/off via --denoise);
    // tuning them further is a natural follow-up once --denoise itself has
    // seen real use.
    baseOptions.nlmWidth = 3.0f;
    baseOptions.nlmFalloff = 10.0f;

    // Compose the manual exposure multiplier above with any physically-
    // derived one from the camera's fStop/shutterSpeed/iso (see
    // Camera::ComputeExposureMultiplier) - this is 1.0 (no-op) unless the
    // scene's camera block set all three, so this is a no-op for every
    // scene that doesn't opt into physical exposure.
    baseOptions.exposure *= scene.camera->ComputeExposureMultiplier();

    // Render-settings-map / multi-product batch pipeline
    // (docs/PRODUCTION_READINESS.md's Tier 3 finding): a scene file's
    // repeatable render_settings { } blocks (if any) drive N output
    // products, possibly across multiple resolutions/sample counts grouped
    // into separate RenderPasses; with none, fall back to exactly the
    // single-pass/single-product behavior every pre-existing invocation
    // (positional args + --aovs/--view-transform) already had.
    RenderSettings settings;
    const std::vector<RenderSettingsDefinition> &renderSettingsDefs = parser.GetRenderSettingsDefs();
    if (!renderSettingsDefs.empty())
    {
        MIRAGE_LOG_DEBUG("Debug: Building render settings from {} render_settings block(s)", renderSettingsDefs.size());
        MIRAGE_LOG_DEBUG(
            "Debug: scene file has render_settings blocks - positional <output_image> arg '{}' is ignored", outputFile);

        settings = BuildRenderSettingsFromDefs(renderSettingsDefs, baseOptions, width, height, samples);
        if (settings.empty())
        {
            MIRAGE_LOG_ERROR("Failed to build render settings from scene file's render_settings blocks");
            delete renderer;
            return 1;
        }
    }
    else
    {
        settings = MakeLegacyRenderSettings(baseOptions, outputFile, aovMask, viewTransform);
    }

    for (RenderPass &pass : settings)
    {
        int passWidth = pass.options.width;
        int passHeight = pass.options.height;

        // --denoise needs albedo/normal as cross-bilateral guide buffers
        // (mirage/filter/NLM.h) regardless of whether any product in this
        // pass actually asked for them as AOV output - requested here so
        // Render() populates them, but product.aovMask (unmodified) is what
        // WriteRenderProduct uses to decide what to actually write to disk,
        // so a guide-only albedo/normal never leaks into a file the user
        // didn't ask for.
        uint32_t guideMask = enableDenoise ? (kAovAlbedo | kAovNormal) : 0u;
        pass.options.aovMask |= guideMask;

        MIRAGE_LOG_DEBUG("Rendering pass ({}x{}, {} samples, {} product(s))", passWidth, passHeight, pass.options.maxSamples,
                          pass.products.size());

        std::vector<Color> outputImage(static_cast<size_t>(passWidth) * static_cast<size_t>(passHeight), Color(0.0f));

        // AOV backing storage - only allocated for buffers this pass
        // actually needs (aovMask's set bits, including any guide-only bits
        // just added above), so a beauty-only pass allocates nothing extra,
        // matching every pre-existing call site's behavior.
        AovBuffers aovs;
        std::vector<Color> aovDepthBuf, aovNormalBuf, aovPrimIdBuf, aovAlbedoBuf;
        if (pass.options.aovMask & kAovDepth)
        {
            aovDepthBuf.assign(outputImage.size(), Color(0.0f));
            aovs.depth = aovDepthBuf.data();
        }
        if (pass.options.aovMask & kAovNormal)
        {
            aovNormalBuf.assign(outputImage.size(), Color(0.0f));
            aovs.normal = aovNormalBuf.data();
        }
        if (pass.options.aovMask & kAovPrimId)
        {
            aovPrimIdBuf.assign(outputImage.size(), Color(0.0f));
            aovs.primId = aovPrimIdBuf.data();
        }
        if (pass.options.aovMask & kAovAlbedo)
        {
            aovAlbedoBuf.assign(outputImage.size(), Color(0.0f));
            aovs.albedo = aovAlbedoBuf.data();
        }

        MIRAGE_LOG_DEBUG("Debug: Starting rendering");
        try
        {
            renderer->Render(*scene.camera, pass.options, outputImage.data(), pass.options.aovMask != 0 ? &aovs : nullptr);
            MIRAGE_LOG_DEBUG("Debug: Rendering completed");
        }
        catch (const std::exception &e)
        {
            MIRAGE_LOG_ERROR("Exception during rendering: {}", e.what());
            delete renderer;
            return 1;
        }
        catch (...)
        {
            MIRAGE_LOG_ERROR("Unknown exception during rendering");
            delete renderer;
            return 1;
        }

        if (enableDenoise)
        {
            MIRAGE_LOG_DEBUG("Debug: Denoising pass output");
            std::vector<Color> denoised(outputImage.size());
            // NonLocalMeansFilter's radius is an int pixel count;
            // Options::nlmWidth is a float for consistency with the rest of
            // Options (see its doc comment) - round rather than truncate,
            // and floor at 1 (radius 0 would mean "no neighborhood at all").
            int radius = std::max(1, static_cast<int>(std::lround(pass.options.nlmWidth)));
            NonLocalMeansFilter(outputImage.data(), denoised.data(), passWidth, passHeight, pass.options.nlmFalloff, radius,
                                 aovs.albedo, aovs.normal);
            outputImage = std::move(denoised);
        }

        bool anyProductFailed = false;
        for (const RenderProduct &product : pass.products)
        {
            if (!WriteRenderProduct(product, outputImage.data(), aovs, passWidth, passHeight, pass.options.exposure))
                anyProductFailed = true;
        }
        if (anyProductFailed)
        {
            delete renderer;
            return 1;
        }
    }

    // Clean up
    MIRAGE_LOG_DEBUG("Debug: Cleaning up");
    delete renderer;

    return 0;
}
