cmake_minimum_required(VERSION 3.15)

# Add the executable
add_executable(scene_renderer SceneRenderer.cpp)

# Include directories
target_include_directories(scene_renderer PRIVATE
    ${CMAKE_SOURCE_DIR}
)

# Link with Mirage library
target_link_libraries(scene_renderer PRIVATE
    Mirage
)

# scene_renderer dynamically depends on the Slang runtime dylibs/.so's that
# kernels/CMakeLists.txt installs into lib/ alongside this binary's bin/ - set
# an install-relative rpath so the installed executable can find them without
# requiring (DY)LD_LIBRARY_PATH juggling. Without this, CMake clears the
# build-tree rpath (which points into the build directory) at install time and
# the binary fails to launch once relocated.
set_target_properties(scene_renderer PROPERTIES
    INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path/../lib,$ORIGIN/../lib>"
)

# Install the executable
install(TARGETS scene_renderer
    RUNTIME DESTINATION bin
)
