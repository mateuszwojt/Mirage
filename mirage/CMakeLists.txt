# Find OpenMP
find_package(OpenMP)

# Vulkan/MoltenVK is the only GPU-capable render path (CUDA has been removed -
# it was never wired into this build and doesn't run on macOS anyway).
find_package(Vulkan REQUIRED)

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
)

set_target_properties(Mirage PROPERTIES
    VERSION ${PROJECT_VERSION}
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Release>:Release>"
    INSTALL_RPATH "$ORIGIN"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(Mirage PRIVATE OpenMP::OpenMP_CXX)
endif()

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
