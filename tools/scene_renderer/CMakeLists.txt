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

# Install the executable
install(TARGETS scene_renderer
    RUNTIME DESTINATION bin
)
