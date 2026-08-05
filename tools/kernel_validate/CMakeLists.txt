cmake_minimum_required(VERSION 3.15)

add_executable(kernel_validate KernelValidate.cpp)

target_include_directories(kernel_validate PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tools/scene_renderer # vendored stb_image_write.h
)

# spdlog::spdlog_header_only: see tools/scene_renderer/CMakeLists.txt's comment - same
# reasoning applies here (KernelValidate.cpp includes mirage/utils/Log.h
# directly, and Mirage links spdlog PRIVATE).
target_link_libraries(kernel_validate PRIVATE
    Mirage
    spdlog::spdlog_header_only
)
# See mirage/utils/Log.h's comment on MIRAGE_LOG_TRACE/DEBUG.
target_compile_definitions(kernel_validate PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

# CPU-vs-GPU kernel parity harness (see the file header comment for scope);
# registered with CTest so it aggregates under one `ctest` entry point
# alongside tests/'s unit/golden suites. CI keeps its own invocation of this
# binary non-blocking (see .github/workflows/build-release.yml's "Run
# kernel_validate (informational)" step) since GitHub-hosted runners' software
# Vulkan ICDs can diverge numerically from the CPU reference on shading, not
# just crash/build regressions - this add_test() registration doesn't change
# that, it just makes `ctest -R kernel_validate` a discoverable local command.
add_test(NAME kernel_validate COMMAND kernel_validate)
