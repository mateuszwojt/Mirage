cmake_minimum_required(VERSION 3.15)

add_executable(kernel_validate KernelValidate.cpp)

target_include_directories(kernel_validate PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tools/scene_renderer # vendored stb_image_write.h
)

target_link_libraries(kernel_validate PRIVATE
    Mirage
)
