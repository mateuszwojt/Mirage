#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cctype>
#include <filesystem>

// Mirage includes
#include "mirage/core/Scene.h"
#include "mirage/prims/Primitive.h"
#include "mirage/shaders/Material.h"
#include "mirage/shaders/TextureLoader.h"
#include "mirage/utils/MathUtils.h"
#include "mirage/utils/Util.h"
#include "mirage/camera/Camera.h"
// #include "mirage/camera/PerspectiveCamera.h"
#include "mirage/math/Vec3.h"
#include "mirage/math/Transform.h"
#include "mirage/core/Renderer.h"
#include "mirage/math/Color.h"

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

    std::cout << "Debug: Split '" << s << "' into " << tokens.size() << " tokens" << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "Debug: Token " << i << ": '" << tokens[i] << "'" << std::endl;
    }

    return tokens;
}

bool startsWith(const std::string &str, const std::string &prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
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

    void setAlbedoMapPath(const std::string &path) { albedoMapPath = path; }
    void setRoughnessMapPath(const std::string &path) { roughnessMapPath = path; }
    void setMetallicMapPath(const std::string &path) { metallicMapPath = path; }

    const std::string &getAlbedoMapPath() const { return albedoMapPath; }
    const std::string &getRoughnessMapPath() const { return roughnessMapPath; }
    const std::string &getMetallicMapPath() const { return metallicMapPath; }

    // Texture loading/registration is deliberately not done here - it needs a
    // Scene& (for FindOrAddTexture's dedup) that this class doesn't have, and
    // the .tin file's own directory (for relative path resolution) that only
    // the parser knows. See PrimitiveDefinition::toPrimitive(), which does
    // that resolution and sets Material::xxxTextureIndex after calling this.
    Material toMaterial() const
    {
        std::cout << "Debug: Converting to Material" << std::endl;
        Material mat;

        // Set material properties
        mat.color = color;
        std::cout << "Debug: Setting color: (" << mat.color.x << ", " << mat.color.y << ", " << mat.color.z << ")" << std::endl;

        mat.roughness = roughness;
        std::cout << "Debug: Setting roughness: " << mat.roughness << std::endl;

        mat.metallic = metallic;
        std::cout << "Debug: Setting metallic: " << mat.metallic << std::endl;

        mat.specular = specular;
        std::cout << "Debug: Setting specular: " << mat.specular << std::endl;

        mat.emission = emission;
        std::cout << "Debug: Setting emission: (" << mat.emission.x << ", " << mat.emission.y << ", " << mat.emission.z << ")" << std::endl;

        return mat;
    }

private:
    Vec3 color = Vec3(0.82f, 0.67f, 0.16f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float specular = 0.5f;
    Vec3 emission = Vec3(0.0f);
    std::string albedoMapPath;
    std::string roughnessMapPath;
    std::string metallicMapPath;
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

    Primitive toPrimitive(const std::map<std::string, MaterialDefinition> &materials, Scene &scene, const std::string &baseDir) const
    {
        std::cout << "Debug: Creating primitive of type: " << type << std::endl;

        Primitive prim;

        // Set transform
        std::cout << "Debug: Setting transform: position(" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
        prim.startTransform.p = position;
        prim.startTransform.r = rotation;
        prim.startTransform.s = scale;

        prim.endTransform = prim.startTransform;

        // Set type and geometry
        if (type == "sphere") {
            std::cout << "Debug: Setting sphere with radius: " << radius << std::endl;
            prim.type = Type::eSphere;
            prim.sphere.radius = radius;
        }
        else if (type == "plane") {
            std::cout << "Debug: Setting plane with normal: (" << plane[0] << ", " << plane[1] << ", " << plane[2] << ", " << plane[3] << ")" << std::endl;
            prim.type = Type::ePlane;
            prim.plane.plane[0] = plane[0];
            prim.plane.plane[1] = plane[1];
            prim.plane.plane[2] = plane[2];
            prim.plane.plane[3] = plane[3];
        }

        // Set material
        std::cout << "Debug: Setting material: " << materialName << std::endl;
        Material mat;
        auto it = materials.find(materialName);
        if (it != materials.end()) {
            mat = it->second.toMaterial();
            std::cout << "Debug: Material found and set" << std::endl;

            // Resolve + load texture maps, relative to the .tin file's own
            // directory (not the process CWD). A missing/broken texture file
            // logs via LoadTextureFromFile's own stderr message and falls
            // back to the material's scalar/color field (xxxTextureIndex
            // stays -1) rather than failing the whole scene load.
            auto resolveAndLoad = [&](const std::string &relPath, TextureColorSpace space) -> int {
                if (relPath.empty())
                    return -1;
                std::string resolvedPath = baseDir.empty() ? relPath : (std::filesystem::path(baseDir) / relPath).string();
                auto tex = LoadTextureFromFile(resolvedPath, space);
                if (!tex)
                    return -1;
                return scene.FindOrAddTexture(resolvedPath, std::move(tex));
            };

            mat.albedoTextureIndex = resolveAndLoad(it->second.getAlbedoMapPath(), TextureColorSpace::eSRGB);
            mat.roughnessTextureIndex = resolveAndLoad(it->second.getRoughnessMapPath(), TextureColorSpace::eLinear);
            mat.metallicTextureIndex = resolveAndLoad(it->second.getMetallicMapPath(), TextureColorSpace::eLinear);
        }
        else {
            std::cout << "Debug: Material not found, using default" << std::endl;
        }
        // FindOrAddMaterial dedups by name, so every primitive referencing the
        // same material block shares one Scene::materials entry rather than
        // getting its own copy.
        prim.materialIndex = scene.FindOrAddMaterial(materialName, std::make_unique<Material>(mat));

        // Set light samples if it's a light
        if (mat.emission.x > 0 ||
            mat.emission.y > 0 ||
            mat.emission.z > 0) {
            std::cout << "Debug: Setting as light with samples: 1" << std::endl;
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
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }

        baseDir = std::filesystem::path(filename).parent_path().string();

        std::cout << "Debug: File opened successfully: " << filename << std::endl;

        std::string line;
        std::string currentSection;
        std::string currentMaterial;
        PrimitiveDefinition currentPrimitive;
        bool inBlock = false;
        bool waitingForBlockStart = false;

        while (std::getline(file, line)) {
            // Trim whitespace
            line = trim(line);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            std::cout << "Debug: Processing line: " << line << std::endl;

            // Check if we're waiting for a block start
            if (waitingForBlockStart) {
                if (line == "{") {
                    waitingForBlockStart = false;
                    inBlock = true;
                    std::cout << "Debug: Block start found" << std::endl;
                    continue;
                }
            }

            // Check for section start/end
            if (line == "{") {
                // Block start
                inBlock = true;
                std::cout << "Debug: Block start found" << std::endl;
            } 
            else if (line == "}") {
                // Block end
                inBlock = false;
                
                if (currentSection == "primitive") {
                    // Add the completed primitive to the list
                    std::cout << "Debug: Adding primitive to list, type: " << currentPrimitive.getType() << std::endl;
                    primitives.push_back(currentPrimitive);
                    currentPrimitive = PrimitiveDefinition();
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
                        std::cout << "Debug: Starting material section: " << currentMaterial << std::endl;
                        
                        // Initialize material with defaults
                        materials[currentMaterial] = MaterialDefinition();
                        waitingForBlockStart = true;
                    } 
                    else if (parts[0] == "primitive") {
                        currentSection = "primitive";
                        std::cout << "Debug: Starting primitive section" << std::endl;
                        
                        // Initialize new primitive
                        currentPrimitive = PrimitiveDefinition();
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
            }
        }
        
        // If we're still in a block at the end of the file, handle it
        if (inBlock && currentSection == "primitive") {
            std::cout << "Debug: Adding final primitive to list" << std::endl;
            primitives.push_back(currentPrimitive);
        }
        
        std::cout << "Debug: Parsing complete. Found " << materials.size() << " materials and " << primitives.size() << " primitives" << std::endl;
        
        return true;
    }

    void setupScene(Scene &scene)
    {
        // Clear existing scene
        std::cout << "Debug: Clearing existing scene" << std::endl;
        scene.Clear();

        // Add primitives
        std::cout << "Debug: Adding " << primitives.size() << " primitives to scene" << std::endl;
        for (const auto &primDef : primitives)
        {
            try
            {
                Primitive prim = primDef.toPrimitive(materials, scene, baseDir);
                std::cout << "Debug: Adding primitive of type " << (prim.type == Type::eSphere ? "sphere" : "plane") << std::endl;
                scene.AddPrimitive(prim);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Exception while adding primitive: " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "Unknown exception while adding primitive" << std::endl;
            }
        }

        // Build the scene (BVH, etc.)
        std::cout << "Debug: Building scene (BVH)" << std::endl;
        try
        {
            scene.Build();
            std::cout << "Debug: Scene built successfully" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception while building scene: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception while building scene" << std::endl;
        }
    }

private:
    void parseMaterialProperty(const std::string &materialName, const std::vector<std::string> &parts)
    {
        if (parts.empty()) return;

        std::string property = parts[0];
        std::cout << "Debug: Parsing material property: " << property << " for material: " << materialName << std::endl;

        try {
            if (property == "color" && parts.size() >= 4) {
                float r = std::stof(parts[1]);
                float g = std::stof(parts[2]);
                float b = std::stof(parts[3]);
                std::cout << "Debug: Setting color: (" << r << ", " << g << ", " << b << ")" << std::endl;
                materials[materialName].setColor(r, g, b);
            }
            else if (property == "roughness" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                std::cout << "Debug: Setting roughness: " << value << std::endl;
                materials[materialName].setRoughness(value);
            }
            else if (property == "metallic" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                std::cout << "Debug: Setting metallic: " << value << std::endl;
                materials[materialName].setMetallic(value);
            }
            else if (property == "specular" && parts.size() >= 2) {
                float value = std::stof(parts[1]);
                std::cout << "Debug: Setting specular: " << value << std::endl;
                materials[materialName].setSpecular(value);
            }
            else if (property == "emission" && parts.size() >= 4) {
                float r = std::stof(parts[1]);
                float g = std::stof(parts[2]);
                float b = std::stof(parts[3]);
                std::cout << "Debug: Setting emission: (" << r << ", " << g << ", " << b << ")" << std::endl;
                materials[materialName].setEmission(r, g, b);
            }
            // Paths with spaces aren't supported - this parser's tokenizer
            // (split()) splits on any space, same pre-existing limitation as
            // every other multi-token property above.
            else if (property == "albedoMap" && parts.size() >= 2) {
                std::cout << "Debug: Setting albedoMap: " << parts[1] << std::endl;
                materials[materialName].setAlbedoMapPath(parts[1]);
            }
            else if (property == "roughnessMap" && parts.size() >= 2) {
                std::cout << "Debug: Setting roughnessMap: " << parts[1] << std::endl;
                materials[materialName].setRoughnessMapPath(parts[1]);
            }
            else if (property == "metallicMap" && parts.size() >= 2) {
                std::cout << "Debug: Setting metallicMap: " << parts[1] << std::endl;
                materials[materialName].setMetallicMapPath(parts[1]);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception while parsing material property: " << e.what() << std::endl;
        }
    }

    void parsePrimitiveProperty(PrimitiveDefinition& primitive, const std::vector<std::string>& parts) {
        if (parts.empty()) return;

        std::string property = parts[0];
        std::cout << "Debug: Parsing primitive property: " << property << std::endl;

        try {
            if (property == "type" && parts.size() >= 2) {
                std::cout << "Debug: Setting primitive type: " << parts[1] << std::endl;
                primitive.setType(parts[1]);
            }
            else if (property == "material" && parts.size() >= 2) {
                std::cout << "Debug: Setting primitive material: " << parts[1] << std::endl;
                primitive.setMaterialName(parts[1]);
            }
            else if (property == "position" && parts.size() >= 4) {
                float x = std::stof(parts[1]);
                float y = std::stof(parts[2]);
                float z = std::stof(parts[3]);
                std::cout << "Debug: Setting position: (" << x << ", " << y << ", " << z << ")" << std::endl;
                primitive.setPosition(x, y, z);
            }
            else if (property == "rotation" && parts.size() >= 5) {
                float x = std::stof(parts[1]);
                float y = std::stof(parts[2]);
                float z = std::stof(parts[3]);
                float w = std::stof(parts[4]);
                std::cout << "Debug: Setting rotation: (" << x << ", " << y << ", " << z << ", " << w << ")" << std::endl;
                primitive.setRotation(x, y, z, w);
            }
            else if (property == "scale" && parts.size() >= 2) {
                float s = std::stof(parts[1]);
                std::cout << "Debug: Setting scale: " << s << std::endl;
                primitive.setScale(s);
            }
            else if (property == "radius" && parts.size() >= 2) {
                float r = std::stof(parts[1]);
                std::cout << "Debug: Setting radius: " << r << std::endl;
                primitive.setRadius(r);
            }
            else if (property == "plane" && parts.size() >= 5) {
                float a = std::stof(parts[1]);
                float b = std::stof(parts[2]);
                float c = std::stof(parts[3]);
                float d = std::stof(parts[4]);
                std::cout << "Debug: Setting plane: (" << a << ", " << b << ", " << c << ", " << d << ")" << std::endl;
                primitive.setPlane(a, b, c, d);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception while parsing primitive property: " << e.what() << std::endl;
        }
    }

    std::map<std::string, MaterialDefinition> materials;
    std::vector<PrimitiveDefinition> primitives;
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
        std::cerr << "Failed to save EXR: " << (err ? err : "unknown error") << std::endl;
        if (err)
            FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

// Main function
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <scene_file> <output_image> [width] [height] [samples]" << std::endl;
        return 1;
    }

    std::string sceneFile = argv[1];
    std::string outputFile = argv[2];

    // Default rendering parameters
    int width = 1280;
    int height = 720;
    int samples = 64;

    // Parse optional parameters
    if (argc > 3)
        width = std::stoi(argv[3]);
    if (argc > 4)
        height = std::stoi(argv[4]);
    if (argc > 5)
        samples = std::stoi(argv[5]);

    std::cout << "Debug: Starting to parse scene file: " << sceneFile << std::endl;

    // Parse the scene file
    SceneParser parser;
    if (!parser.parse(sceneFile))
    {
        std::cerr << "Failed to parse scene file." << std::endl;
        return 1;
    }

    std::cout << "Debug: Scene file parsed successfully" << std::endl;

    // Create and setup the scene
    Scene scene;
    std::cout << "Debug: Setting up scene" << std::endl;
    parser.setupScene(scene);

    std::cout << "Debug: Scene setup complete, primitives count: " << scene.primitives.size() << std::endl;

    // Create a default camera if none was specified in the scene
    if (!scene.camera)
    {
        std::cout << "Debug: Creating default camera" << std::endl;
        auto camera = std::make_unique<Camera>();
        camera->position = Vec3(0.0f, 2.0f, 5.0f);
        // camera->lookAt = Vec3(0.0f, 0.0f, 0.0f);
        // camera->up = Vec3(0.0f, 1.0f, 0.0f);
        camera->fov = 45.0f;
        // camera->aspect = float(width) / float(height);

        scene.camera = std::move(camera);
    }

    // Render the scene
    std::cout << "Rendering scene to " << outputFile << " (" << width << "x" << height << ", " << samples << " samples)" << std::endl;

    // Create renderer
    std::cout << "Debug: Creating renderer" << std::endl;
    Renderer *renderer = CreateCpuRenderer(&scene);
    if (!renderer)
    {
        std::cerr << "Failed to create renderer" << std::endl;
        return 1;
    }

    // Setup rendering options
    std::cout << "Debug: Setting up rendering options" << std::endl;
    Options options;
    options.width = width;
    options.height = height;
    options.maxSamples = samples;
    options.maxDepth = 5;
    options.mode = RenderMode::ePathTrace;
    options.type = RenderType::eCpu;
    options.filter = Filter(FilterType::eFilterGaussian, 1.0f, 2.0f);
    options.exposure = 1.0f;
    options.clamp = 10.0f;
    options.enableDOF = false;

    // Allocate memory for output image
    std::cout << "Debug: Allocating memory for output image" << std::endl;
    std::vector<Color> outputImage(width * height, Color(0.0f));

    // Render the scene
    std::cout << "Debug: Starting rendering" << std::endl;
    try
    {
        renderer->Render(*scene.camera, options, outputImage.data());
        std::cout << "Debug: Rendering completed" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception during rendering: " << e.what() << std::endl;
        delete renderer;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception during rendering" << std::endl;
        delete renderer;
        return 1;
    }

    // Save the image
    std::cout << "Debug: Saving image" << std::endl;
    std::string extension = outputFile.substr(outputFile.find_last_of(".") + 1);
    bool success = false;

    if (extension == "exr")
    {
        // Full float32, untonemapped output - skip the 8-bit LDR conversion
        // below entirely, tonemapping/quantization are LDR-only concerns.
        std::cout << "Debug: Writing float EXR output" << std::endl;
        success = WriteExr(outputFile, outputImage.data(), width, height);
    }
    else
    {
        // Convert the rendered image to 8-bit RGB for saving
        std::cout << "Debug: Converting image for saving" << std::endl;
        std::vector<unsigned char> imageData(width * height * 3);

        for (int i = 0; i < width * height; i++)
        {
            // Apply tone mapping and convert to sRGB
            Color pixel = ToneMap(outputImage[i], options.exposure);

            // Convert to 8-bit RGB
            int r = int(255.0f * Clamp(pixel.x, 0.0f, 1.0f));
            int g = int(255.0f * Clamp(pixel.y, 0.0f, 1.0f));
            int b = int(255.0f * Clamp(pixel.z, 0.0f, 1.0f));

            // Store in the output buffer (RGB format)
            imageData[i * 3 + 0] = (unsigned char)r;
            imageData[i * 3 + 1] = (unsigned char)g;
            imageData[i * 3 + 2] = (unsigned char)b;
        }

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
            // appending, so e.g. "out.foo" becomes "out.png", not
            // "out.foo.png".
            std::cerr << "Warning: unrecognized output extension '" << extension << "', defaulting to PNG" << std::endl;
            size_t dotPos = outputFile.find_last_of(".");
            outputFile = (dotPos != std::string::npos) ? outputFile.substr(0, dotPos) + ".png" : outputFile + ".png";
            success = stbi_write_png(outputFile.c_str(), width, height, 3, imageData.data(), width * 3) != 0;
        }
    }

    if (success)
    {
        std::cout << "Image saved to " << outputFile << std::endl;
    }
    else
    {
        std::cerr << "Failed to save image to " << outputFile << std::endl;
        return 1;
    }

    // Clean up
    std::cout << "Debug: Cleaning up" << std::endl;
    delete renderer;

    return 0;
}
