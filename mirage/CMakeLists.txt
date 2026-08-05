# Find OpenMP
find_package(OpenMP)

# Vulkan (native on Linux, via MoltenVK on macOS) is the only GPU-capable
# render path (CUDA has been removed - it was never wired into this build).
find_package(Vulkan REQUIRED)

# spdlog: structured logging (mirage/utils/Log.h), resolving
# docs/PRODUCTION_READINESS.md's Tier 3 "no logging framework" finding.
# Pinned by tag, fetched reproducibly rather than vendored - same pattern as
# the Slang/slang-rhi pins in kernels/CMakeLists.txt. Declared here (not in
# tests/CMakeLists.txt) because it's a real runtime dependency of the Mirage
# library itself, not just of the test suite; tools/scene_renderer and
# tools/kernel_validate additionally link spdlog::spdlog_header_only directly (see their
# own CMakeLists.txt) since Mirage links it PRIVATE and CMake doesn't
# propagate a PRIVATE dependency's include paths to consumers - the imported
# spdlog::spdlog_header_only target itself is visible globally once FetchContent_MakeAvailable
# runs here, since add_subdirectory(mirage) is processed before
# add_subdirectory(tools) in the top-level CMakeLists.txt.
include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)

add_subdirectory(kernels)

# build static library
add_library(Mirage STATIC
    camera/Camera.cpp
    core/Renderer.cpp
    core/Scene.cpp
    core/HydraCompatibility.cpp
    core/VulkanRenderer.cpp
    filter/NLM.cpp
    prims/Mesh.cpp
    shaders/TextureLoader.cpp
    utils/Log.cpp
    utils/CrashHandler.cpp
)

set_target_properties(Mirage PROPERTIES
    VERSION ${PROJECT_VERSION}
    # CMake's MSVC_RUNTIME_LIBRARY only recognizes four literal values
    # (MultiThreaded[Debug][DLL]) - the conditional suffix must be "Debug",
    # never "Release". The previous "MultiThreadedDLL$<$<CONFIG:Release>:Release>"
    # evaluated to the non-existent "MultiThreadedDLLRelease" on any
    # single-config MSVC build with CMAKE_BUILD_TYPE=Release (e.g. CI's
    # windows-x86_64 job, the first real MSVC build this repo has run),
    # failing CMake's generate step. This never surfaced on macOS/Linux since
    # non-MSVC toolchains silently ignore MSVC_RUNTIME_LIBRARY.
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    INSTALL_RPATH "$ORIGIN"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(Mirage PRIVATE OpenMP::OpenMP_CXX)
endif()

# $<BUILD_INTERFACE:...>: spdlog_header_only is a plain FetchContent'd
# INTERFACE target with no install()/export rules of its own (same class of
# problem slang-rhi's target has below, but simpler to resolve here - since
# it's header-only, spdlog's code is fully inlined into Mirage's own compiled
# .cpp files at build time, so an installed libMirage.a has no leftover
# spdlog symbol requirement for downstream consumers to satisfy; unlike
# slang-rhi there is no compiled artifact that also needs bundling into the
# install for INSTALL_INTERFACE). Without this generator-expression guard,
# `install(EXPORT MirageTargets ...)` below fails configure with "requires
# target spdlog_header_only that is not in any export set".
target_link_libraries(Mirage PRIVATE $<BUILD_INTERFACE:spdlog::spdlog_header_only>)
# See mirage/utils/Log.h's comment on MIRAGE_LOG_TRACE/DEBUG: without this,
# spdlog's default compile-time active level (info) would silently no-op
# those macros regardless of Mirage::Logging::Init()'s runtime level.
target_compile_definitions(Mirage PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

# PUBLIC: consumers linking Mirage::Mirage (e.g. hdMirage) also need the Vulkan
# loader symbols, since VulkanRenderer.h's interface is reachable from Renderer.h.
#
# slang-rhi is a real in-tree CMake target for local (BUILD_INTERFACE) builds -
# e.g. Mirage's own tools/kernel_validate - but it has no install()/export
# support of its own, so an *installed* Mirage package instead links against
# Mirage::slang-rhi-bundled, an IMPORTED target that MirageConfig.cmake defines
# from the plain library files installed by kernels/CMakeLists.txt. See that
# file and MirageConfig.cmake.in for the full rationale.
target_link_libraries(Mirage PUBLIC
    Vulkan::Vulkan
    $<BUILD_INTERFACE:slang-rhi>
    $<INSTALL_INTERFACE:Mirage::slang-rhi-bundled>
)
target_link_libraries(Mirage PRIVATE MirageKernels)

# Any binary linking Mirage::Mirage (e.g. hdMirage.dylib) needs the Slang
# runtime dylibs (libslang.dylib and friends) resolvable at load time.
# Propagate an rpath pointing at Mirage's own installed lib/ dir so consumers
# get this for free, without DYLD_LIBRARY_PATH juggling.
include(GNUInstallDirs)
set_target_properties(Mirage PROPERTIES
    INTERFACE_LINK_OPTIONS "$<INSTALL_INTERFACE:-Wl,-rpath,${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}>"
)

# Specify the include directories
target_include_directories(Mirage PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install the library into the lib directory, exported as the Mirage::Mirage
# target so downstream consumers can `find_package(Mirage CONFIG)`.
install(TARGETS Mirage
    EXPORT MirageTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install public header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MirageConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/MirageConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MirageConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Mirage"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MirageConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MirageConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Mirage"
)
install(EXPORT MirageTargets
    NAMESPACE Mirage::
    FILE MirageTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Mirage"
)
