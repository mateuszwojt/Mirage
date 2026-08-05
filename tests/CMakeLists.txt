# Mirage's unit test suite, bootstrapped per docs/PRODUCTION_READINESS.md's
# Tier 3 "no scene-level test coverage" finding. doctest (not
# GTest/Catch2) matches this repo's existing preference for small,
# single-header, FetchContent-pinned dependencies (see mirage/thirdparty/
# and mirage/kernels/CMakeLists.txt's Slang/slang-rhi pins) over a larger
# framework requiring its own build.
#
# This directory is deliberately separate from mirage/kernels' FetchContent
# usage and gated behind MIRAGE_BUILD_TESTS (see top-level CMakeLists.txt) so
# an installed `find_package(Mirage)` consumer never pulls in test-only deps.

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.5.3
)
FetchContent_MakeAvailable(doctest)

add_executable(mirage_unit_tests
    unit/TestMain.cpp
    unit/ColorTest.cpp
    unit/AovTest.cpp
    unit/AovRenderTest.cpp
    unit/LogTest.cpp
    unit/ViewTransformTest.cpp
    unit/NLMTest.cpp
    unit/RenderSettingsTest.cpp
)

target_include_directories(mirage_unit_tests PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(mirage_unit_tests PRIVATE
    Mirage
    doctest::doctest
    spdlog::spdlog_header_only
)
# See mirage/utils/Log.h's comment on MIRAGE_LOG_TRACE/DEBUG - same reasoning
# as scene_renderer/kernel_validate's CMakeLists.txt.
target_compile_definitions(mirage_unit_tests PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")
doctest_discover_tests(mirage_unit_tests)
