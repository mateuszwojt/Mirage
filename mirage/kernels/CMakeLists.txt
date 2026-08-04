# Slang kernel toolchain + slang-rhi runtime, fetched reproducibly instead of
# vendored, per the "Mirage: Slang + Vulkan (MoltenVK) Renderer Refactor" plan.
#
# Two external, non-find_package-able dependencies are pulled in here:
#   - the prebuilt Slang compiler/runtime distribution (gives us `slangc` plus the
#     headers/libs slang-rhi links against for shader reflection)
#   - slang-rhi itself (the Slang project's GPU abstraction library), built from
#     source against the same Slang distribution, with only the Vulkan/Metal/WGPU
#     backends it builds by default on Darwin.
#
# Both are pinned (release tag / commit sha) rather than tracking a branch -
# slang-rhi does not yet publish tagged releases, so a commit sha is the pin.

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

set(MIRAGE_SLANG_VERSION "2026.14")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(MIRAGE_SLANG_DIST_URL "https://github.com/shader-slang/slang/releases/download/v${MIRAGE_SLANG_VERSION}/slang-${MIRAGE_SLANG_VERSION}-macos-aarch64.tar.gz")
        set(MIRAGE_SLANG_DIST_HASH "SHA256=a5c82b66eac051bc78a06d2d8becf0e1003310f9947d9b645d7da0a1e48545c1")
    else()
        message(FATAL_ERROR "Mirage's Slang/Vulkan kernel path does not yet have a pinned Slang distribution for macOS ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
else()
    message(FATAL_ERROR "Mirage's Slang/Vulkan kernel path is currently only wired up for macOS (MoltenVK); ${CMAKE_SYSTEM_NAME} is not yet supported here")
endif()

FetchContent_Declare(
    mirage_slang_dist
    URL "${MIRAGE_SLANG_DIST_URL}"
    URL_HASH "${MIRAGE_SLANG_DIST_HASH}"
    SOURCE_SUBDIR "_does_not_exist_" # this is a prebuilt binary distribution, not a CMake project
)
FetchContent_MakeAvailable(mirage_slang_dist)

set(MIRAGE_SLANGC "${mirage_slang_dist_SOURCE_DIR}/bin/slangc" CACHE FILEPATH "Path to the slangc compiler used to build Mirage's kernels" FORCE)

# --- slang-rhi ---------------------------------------------------------------

set(SLANG_RHI_COMMIT "1afb8384634bc8e9fc7ff2d75f8a0dbec5502f38")

set(SLANG_RHI_FETCH_SLANG OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_SLANG_INCLUDE_DIR "${mirage_slang_dist_SOURCE_DIR}/include" CACHE STRING "" FORCE)
set(SLANG_RHI_SLANG_BINARY_DIR "${mirage_slang_dist_SOURCE_DIR}" CACHE STRING "" FORCE)
set(SLANG_RHI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_TESTS_WITH_GLFW OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    slang_rhi
    GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
    GIT_TAG "${SLANG_RHI_COMMIT}"
)
FetchContent_MakeAvailable(slang_rhi)

# --- .slang -> SPIR-V, compiled in as a generated header ---------------------
#
# Build-time compilation (not runtime Slang-API compilation) so shader errors
# surface during the C++ build, and so the shipped static library doesn't need
# to bundle the full Slang compiler - see the plan's "Host-side VulkanRenderer"
# section for the rationale.

set(MIRAGE_SLANG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/slang")
set(MIRAGE_SLANG_COMMON_DIR "${MIRAGE_SLANG_DIR}/common")
set(MIRAGE_SLANG_MODULE_DIRS
    "${MIRAGE_SLANG_DIR}/common"
    "${MIRAGE_SLANG_DIR}/geometry"
    "${MIRAGE_SLANG_DIR}/sampling"
    "${MIRAGE_SLANG_DIR}/scene"
    "${MIRAGE_SLANG_DIR}/shading"
    "${MIRAGE_SLANG_DIR}/integrator"
    "${MIRAGE_SLANG_DIR}/spike"
)
set(MIRAGE_SPIRV_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv")
file(MAKE_DIRECTORY "${MIRAGE_SPIRV_OUT_DIR}")

set(MIRAGE_SLANGC_INCLUDE_FLAGS "")
foreach(DIR ${MIRAGE_SLANG_MODULE_DIRS})
    list(APPEND MIRAGE_SLANGC_INCLUDE_FLAGS -I "${DIR}")
endforeach()

file(GLOB_RECURSE MIRAGE_ALL_SLANG_SOURCES "${MIRAGE_SLANG_DIR}/*.slang")

# embed_spirv(<slang-source> <entry-point> <out-var-name> <output-target-name>)
# Compiles <slang-source> to SPIR-V and generates a header exposing the bytecode
# as `static const uint32_t <out-var-name>[]` + `<out-var-name>_size`. This is a
# build-time validation/CI gate (catches compile errors early) - it is NOT what
# VulkanRenderer actually loads at runtime; see the compile definitions below.
function(mirage_embed_spirv SLANG_SOURCE ENTRY_POINT SYMBOL_NAME OUT_TARGET)
    get_filename_component(BASE_NAME ${SLANG_SOURCE} NAME_WE)
    set(SPV_FILE "${MIRAGE_SPIRV_OUT_DIR}/${BASE_NAME}.spv")
    set(HEADER_FILE "${MIRAGE_SPIRV_OUT_DIR}/${BASE_NAME}.spv.h")

    add_custom_command(
        OUTPUT "${SPV_FILE}"
        COMMAND "${MIRAGE_SLANGC}"
                "${SLANG_SOURCE}"
                ${MIRAGE_SLANGC_INCLUDE_FLAGS}
                -target spirv
                -force-glsl-scalar-layout
                -matrix-layout-column-major
                -entry "${ENTRY_POINT}"
                -o "${SPV_FILE}"
        DEPENDS "${SLANG_SOURCE}" ${MIRAGE_ALL_SLANG_SOURCES}
        COMMENT "slangc: compiling ${BASE_NAME}.slang -> SPIR-V"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${HEADER_FILE}"
        COMMAND ${CMAKE_COMMAND}
                -DSPV_FILE=${SPV_FILE}
                -DHEADER_FILE=${HEADER_FILE}
                -DSYMBOL_NAME=${SYMBOL_NAME}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/EmbedSpirv.cmake"
        DEPENDS "${SPV_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/EmbedSpirv.cmake"
        COMMENT "Embedding ${BASE_NAME}.spv as ${HEADER_FILE}"
        VERBATIM
    )

    add_custom_target(${OUT_TARGET} DEPENDS "${HEADER_FILE}")
endfunction()

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/spike/ClearColor.slang"
    main
    g_ClearColorSpirv
    MirageShaders_ClearColor
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/integrator/NormalsMain.slang"
    main
    g_NormalsMainSpirv
    MirageShaders_NormalsMain
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/integrator/PathTraceMain.slang"
    main
    g_PathTraceMainSpirv
    MirageShaders_PathTraceMain
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/integrator/PathTraceTestMain.slang"
    main
    g_PathTraceTestMainSpirv
    MirageShaders_PathTraceTestMain
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/shading/BsdfTestMain.slang"
    main
    g_BsdfTestMainSpirv
    MirageShaders_BsdfTestMain
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/shading/ProbeTestMain.slang"
    main
    g_ProbeTestMainSpirv
    MirageShaders_ProbeTestMain
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/spike/TextureArraySpike.slang"
    main
    g_TextureArraySpikeSpirv
    MirageShaders_TextureArraySpike
)

mirage_embed_spirv(
    "${MIRAGE_SLANG_DIR}/shading/TextureSampleTestMain.slang"
    main
    g_TextureSampleTestMainSpirv
    MirageShaders_TextureSampleTestMain
)

add_custom_target(MirageShaders DEPENDS
    MirageShaders_ClearColor
    MirageShaders_NormalsMain
    MirageShaders_BsdfTestMain
    MirageShaders_ProbeTestMain
    MirageShaders_PathTraceMain
    MirageShaders_PathTraceTestMain
    MirageShaders_TextureArraySpike
    MirageShaders_TextureSampleTestMain
)

add_library(MirageKernels INTERFACE)
# BUILD_INTERFACE only: this points at the build-tree embedded-SPIRV headers,
# needed only to compile Mirage's own .cpp files. Installed consumers never
# see these headers (the SPIRV is already baked into the compiled libMirage.a).
target_include_directories(MirageKernels INTERFACE "$<BUILD_INTERFACE:${MIRAGE_SPIRV_OUT_DIR}>")
add_dependencies(MirageKernels MirageShaders)

# INTERFACE library, no build artifacts of its own - only needs to be part of
# the export set so install(EXPORT MirageTargets) can resolve Mirage's PRIVATE
# link dependency on it.
install(TARGETS MirageKernels EXPORT MirageTargets)

# VulkanRenderer resolves `import`s via these search paths at runtime through
# slang-rhi's Slang session (see plan's "Host-side VulkanRenderer" section - the
# build-time slangc pass above is a validation/CI gate, not the only compile).
#
# This macro gets baked once into VulkanRenderer.cpp's compiled object code, so
# it can't distinguish "used from the build tree" vs. "used from an installed
# Mirage package" at runtime. Instead it lists candidate paths for both: the
# source-tree location (so mirage's own tools/kernel_validate keep working
# without requiring `cmake --install` first) and the installed
# share/mirage/kernels/slang location (installed by this file below, and what
# hdMirage - which never has the mirage source tree available - actually
# needs). slang's module loader simply skips search paths that don't resolve
# an import, so listing both is harmless.
set(MIRAGE_SLANG_SEARCH_PATH_DEFINES "")
foreach(DIR ${MIRAGE_SLANG_MODULE_DIRS})
    list(APPEND MIRAGE_SLANG_SEARCH_PATH_DEFINES "\"${DIR}\"")
endforeach()
foreach(DIR ${MIRAGE_SLANG_MODULE_DIRS})
    get_filename_component(SUBDIR_NAME "${DIR}" NAME)
    list(APPEND MIRAGE_SLANG_SEARCH_PATH_DEFINES "\"${CMAKE_INSTALL_PREFIX}/share/mirage/kernels/slang/${SUBDIR_NAME}\"")
endforeach()
string(REPLACE ";" ", " MIRAGE_SLANG_SEARCH_PATH_LIST "${MIRAGE_SLANG_SEARCH_PATH_DEFINES}")

target_compile_definitions(MirageKernels INTERFACE
    "MIRAGE_SLANG_SEARCH_PATHS={${MIRAGE_SLANG_SEARCH_PATH_LIST}}"
)

# --- Install: bundle slang-rhi's build artifacts, the Slang runtime dylibs,
# and the .slang sources themselves, since none of this is exportable via a
# normal install(EXPORT) (see MirageConfig.cmake.in for the full rationale).

install(DIRECTORY "${MIRAGE_SLANG_DIR}/"
    DESTINATION share/mirage/kernels/slang
    FILES_MATCHING PATTERN "*.slang"
)

install(FILES
    "$<TARGET_FILE:slang-rhi>"
    "$<TARGET_FILE:slang-rhi-resources>"
    "$<TARGET_FILE:slang-rhi-vma>"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

file(GLOB MIRAGE_SLANG_RUNTIME_DYLIBS "${mirage_slang_dist_SOURCE_DIR}/lib/*.dylib")
install(FILES ${MIRAGE_SLANG_RUNTIME_DYLIBS} DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# Best-effort only: slang-rhi's WGPU/Dawn backend is never used by Mirage (the
# Vulkan backend is the only one VulkanRenderer requests), and dawn is not a
# link-time dependency of slang-rhi (its CMake only wires it in via a runtime
# copy-next-to-the-binary step - see slang-rhi's own CMakeLists.txt). Installed
# defensively in case some future slang-rhi init path probes for it eagerly.
if(DEFINED dawn_SOURCE_DIR AND EXISTS "${dawn_SOURCE_DIR}/lib/libdawn.dylib")
    install(FILES "${dawn_SOURCE_DIR}/lib/libdawn.dylib" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
